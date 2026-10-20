from datetime import datetime, timedelta, timezone
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from traid_ai.dashboard import DashboardServer, SessionManager, chart_data, parse_settings
from traid_ai.engine import Engine
from traid_ai.models import Config, D, Quote, stamp
from traid_ai.store import Store, writer_lock


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "paper.sqlite3"
        self.manager = SessionManager(self.path, demo_interval=0.01)
        self.server = DashboardServer(("127.0.0.1", 0), self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.manager.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        body = json.dumps(payload) if payload is not None else None
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        connection.request(method, path, body=body, headers=h)
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        return status, data

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            view = self.manager.view()
            if predicate(view):
                return view
            time.sleep(0.01)
        self.fail("Состояние исполнителя не достигнуто")

    def test_http_start_pause_stop_and_export(self):
        status, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("execution-safety-guards-v1", json.loads(body)["capabilities"])
        token = json.loads(body)["control_token"]
        headers = {"X-Control-Token": token}
        status, body = self.request("POST", "/api/start", {"mode": "demo"}, headers)
        self.assertEqual(status, 200, body)
        session_id = json.loads(body)["session_id"]
        self.wait_for(lambda v: v["report"]["entries"] > 0)
        status, body = self.request("POST", "/api/control", {"session_id": session_id, "action": "pause"}, headers)
        self.assertEqual(status, 200, body)
        paused = self.wait_for(lambda v: v["report"]["status"] == "PAUSED_ENTRIES")
        entries = paused["report"]["entries"]
        time.sleep(0.12)
        self.assertEqual(self.manager.view()["report"]["entries"], entries)
        status, body = self.request("POST", "/api/control", {"session_id": session_id, "action": "stop"}, headers)
        self.assertEqual(status, 200, body)
        view = self.wait_for(lambda v: v["report"]["status"] == "CLOSED")
        self.assertEqual(D(view["report"]["quantity"]), 0)
        self.assertEqual(D(view["points"][-1]["equity"]), D(view["report"]["equity_at_last_quote"]))
        self.assertTrue(view["fills"])
        status, body = self.request("GET", "/api/export?session=" + session_id)
        self.assertEqual(status, 200, body)
        self.assertGreater(len(json.loads(body)["events"]), 10)

    def test_cross_origin_and_missing_token_are_rejected(self):
        status, _ = self.request("POST", "/api/start", {"mode": "demo"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/api/start", {"mode": "demo"},
                                 {"X-Control-Token": self.server.token, "Origin": "https://example.com"})
        self.assertEqual(status, 403)
        status, _ = self.request("GET", "/api/state", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        self.assertFalse(self.manager.running)

    def test_second_launch_cannot_replace_active_worker(self):
        session_id = self.manager.launch(payload={"mode": "demo"})
        with self.assertRaises(ValueError):
            self.manager.launch(payload={"mode": "demo"})
        self.assertEqual(self.manager.active_id, session_id)

    def test_resume_saved_pause_and_keep_original_deadline(self):
        store = Store(self.path)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        engine = Engine.start(store, Config(source="synthetic"), start, start + timedelta(hours=1))
        engine.step(start, Quote("a", stamp(start), D("100"), D("100.1"), "synthetic"), pause_entries=True)
        session_id, deadline = engine.state.session_id, engine.state.deadline_at
        store.close()
        with self.assertRaises(ValueError):
            self.manager.launch(payload={"mode": "demo"})
        self.manager.launch(session_id=session_id)
        view = self.wait_for(lambda v: v["state"] and v["state"]["last_tick_at"] != stamp(start))
        self.assertEqual(view["state"]["status"], "PAUSED_ENTRIES")
        self.assertEqual(view["state"]["deadline_at"], deadline)
        self.manager.control("continue", session_id)
        self.wait_for(lambda v: v["state"]["status"] == "RUNNING")

    def test_pause_keeps_protective_exit(self):
        store = Store(self.path)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        engine = Engine.start(store, Config(source="synthetic"), start, start + timedelta(hours=1))
        try:
            for index in range(6):
                now = start + timedelta(seconds=index * 10)
                price = D("102") if index == 5 else D("100")
                engine.step(now, Quote(str(index), stamp(now), price, price + D(".1"), "synthetic"))
            now = start + timedelta(seconds=60)
            engine.step(now, Quote("pause-exit", stamp(now), D("98"), D("98.1"), "synthetic"), pause_entries=True)
            self.assertEqual(engine.state.status, "PAUSED_ENTRIES")
            self.assertEqual(D(engine.state.quantity), 0)
            self.assertEqual(engine.state.exits, 1)
        finally:
            store.close()

    def test_public_data_failure_leaves_unresolved_and_visible(self):
        store = Store(self.path)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        engine = Engine.start(store, Config(source="kraken-public"), start, start + timedelta(hours=1))
        for index in range(6):
            now = start + timedelta(seconds=index * 10)
            price = D("102") if index == 5 else D("100")
            engine.step(now, Quote(str(index), stamp(now), price, price + D(".1"), "kraken-public"))
        session_id = engine.state.session_id
        store.close()
        with patch("traid_ai.dashboard.public_quote", side_effect=ValueError("offline")):
            self.manager.launch(session_id=session_id)
            view = self.wait_for(lambda v: v["report"] and v["report"]["status"] == "UNRESOLVED")
        self.assertGreater(D(view["report"]["quantity"]), 0)
        self.assertTrue(any(e["kind"] == "DATA_UNAVAILABLE" for e in view["events"]))

    def test_invalid_settings_do_not_start_worker(self):
        for data in ({"mode": "live"}, {"capital": "NaN"}, {"duration_minutes": "1"}, {"risk_percent": "0"}):
            with self.assertRaises(ValueError):
                self.manager.launch(payload=data)
        self.assertFalse(self.manager.running)

    def saved_session(self, *, status="CLOSED", quantity="0", pending=None):
        store = Store(self.path)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        engine = Engine.start(store, Config(source="synthetic"), now, now + timedelta(hours=1))
        engine.state.status = status
        engine.state.quantity = quantity
        engine.state.broker = {"pending": pending}
        store.commit_step(engine.state, "fixture", stamp(now), [("FIXTURE", {})])
        store.save_analysis(engine.state.session_id, "comparison", {"test": True})
        store.close()
        return engine.state.session_id

    def test_delete_archive_removes_all_records_and_resets_selection(self):
        keep = self.saved_session()
        target = self.saved_session()
        self.manager.active_id = target
        status, body = self.request("POST", "/api/sessions/delete", {"session_id": target},
                                    {"X-Control-Token": self.server.token})
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["deleted_id"], target)
        self.assertIsNone(self.manager.active_id)
        view = self.manager.view()
        self.assertEqual(view["selected_id"], keep)
        store = Store(self.path, readonly=True)
        try:
            for table in ("events", "steps", "analyses", "sessions"):
                column = "id" if table == "sessions" else "session_id"
                self.assertEqual(store.db.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (target,)).fetchone()[0], 0)
            self.assertTrue(store.events(keep))
            self.assertEqual(store.analysis(keep, "comparison"), {"test": True})
        finally:
            store.close()
        self.manager.delete_session(keep)
        self.assertIsNone(self.manager.view()["report"])

    def test_delete_rejects_unfinished_position_and_pending_order(self):
        for values in ({"status": "RUNNING"}, {"status": "UNRESOLVED"},
                       {"quantity": "0.00000001"}, {"pending": {"phase": "UNKNOWN"}}, {"pending": {}}):
            sid = self.saved_session(**values)
            with self.assertRaisesRegex(ValueError, "завершённую сессию"):
                self.manager.delete_session(sid)
            self.assertEqual(self.manager.view(sid)["event_count"], 2)

    def test_delete_rejects_execution_maintenance_and_external_writer(self):
        target = self.saved_session()
        with writer_lock(self.path):
            with self.assertRaisesRegex(ValueError, "другим исполнителем"):
                self.manager.delete_session(target)
        self.manager.maintenance["running"] = True
        with self.assertRaises(ValueError):
            self.manager.delete_session(target)
        self.manager.maintenance["running"] = False
        self.manager.launch(payload={"mode": "demo"})
        with self.assertRaises(ValueError):
            self.manager.delete_session(target)
        self.assertEqual(self.manager.view(target)["event_count"], 2)

    def test_delete_is_atomic_on_storage_failure(self):
        target = self.saved_session()
        store = Store(self.path)
        store.db.execute("CREATE TRIGGER deny_delete BEFORE DELETE ON steps BEGIN SELECT RAISE(ABORT, 'fixture'); END")
        store.db.commit()
        store.close()
        status, _ = self.request("POST", "/api/sessions/delete", {"session_id": target},
                                 {"X-Control-Token": self.server.token})
        self.assertEqual(status, 400)
        store = Store(self.path, readonly=True)
        try:
            self.assertEqual(len(store.events(target)), 2)
            self.assertIsNotNone(store.analysis(target, "comparison"))
            self.assertTrue(store.seen(target, "fixture"))
            self.assertEqual(store.load(target)[1].status, "CLOSED")
        finally:
            store.close()

    def test_delete_security_missing_session_and_unknown_id(self):
        target = self.saved_session()
        for headers in ({}, {"X-Control-Token": self.server.token, "Origin": "https://example.com"},
                        {"X-Control-Token": self.server.token, "Host": "evil.example"}):
            self.assertEqual(self.request("POST", "/api/sessions/delete", {"session_id": target}, headers)[0], 403)
        for payload in ({}, {"session_id": []}, {"session_id": "missing"}):
            self.assertEqual(self.request("POST", "/api/sessions/delete", payload, {"X-Control-Token": self.server.token})[0], 400)
        self.assertEqual(self.manager.view(target)["event_count"], 2)

    def test_history_includes_sessions_older_than_first_hundred(self):
        store = Store(self.path)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ids = [Engine.start(store, Config(source="synthetic"), now, now + timedelta(hours=1)).state.session_id
               for _ in range(101)]
        store.close()
        view = self.manager.view(ids[0])
        self.assertEqual(len(view["sessions"]), 101)
        self.assertEqual(view["selected_id"], ids[0])
        self.assertEqual(view["sessions"][-1]["session_id"], ids[0])

    def test_default_selection_ignores_archive_deleted_by_other_manager(self):
        keep = self.saved_session()
        target = self.saved_session()
        self.manager.active_id = target
        other = SessionManager(self.path)
        other.delete_session(target)
        self.assertEqual(self.manager.view()["selected_id"], keep)

    def test_static_page_contains_controls_and_charts(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        for value in (b'id="settings"', b'id="price-chart"', b'id="pause"', b'id="sessions"',
                      b'name="target_percent"',b'name="stop_percent"',b'id="objective-panel"'):
            self.assertIn(value, body)
        target_input=body.split(b'<input id="target"',1)[1].split(b'>',1)[0]
        self.assertNotIn(b'max=',target_input)

    def test_http_soft_goal_and_trailing_stop_are_persisted(self):
        status,body=self.request("GET","/api/state")
        headers={"X-Control-Token":json.loads(body)["control_token"]}
        status,body=self.request("POST","/api/start",dict(mode="demo",duration_minutes="5",
            target_percent="250",stop_percent="0.8"),headers)
        self.assertEqual(status,200,body)
        sid=json.loads(body)["session_id"]
        status,body=self.request("GET","/api/state?session="+sid)
        view=json.loads(body)
        self.assertEqual(view["config"]["exit_policy"],"trailing-stop-v1")
        self.assertEqual(D(view["config"]["target_return"]),D("2.5"))
        self.assertEqual(D(view["config"]["stop_fraction"]),D(".008"))
        self.assertFalse(view["report"]["objective"]["mandatory"])
        self.assertEqual(D(view["report"]["objective"]["target_pnl_usd"]),D("2500"))


if __name__ == "__main__":
    unittest.main()
