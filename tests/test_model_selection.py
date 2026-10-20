from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import http.client
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from traid_ai.dashboard import DashboardServer, SessionManager, parse_settings
from traid_ai.engine import Engine
from traid_ai.evaluation import compare_session
from traid_ai.models import Config, D, Quote, State, stamp, utc_now
from traid_ai.qwen import MODEL, ModelConnection, ModelError, OllamaClient, build_snapshot
from traid_ai.store import Store
from test_alpaca import await_condition
from test_evaluation import dataset, add_decision

NINE = "qwen3.5:9b"
DIGEST = "b"*64


class ProbeModel:
    def __init__(self, model=MODEL):
        self.model = model
        self.snapshots = []
        self.fail = False
        self.release = threading.Event()
        self.release.set()

    def status(self):
        if self.fail:
            raise ModelError("missing model")
        return dict(model=self.model, digest=DIGEST, parameter_size="8.95B" if self.model == NINE else "4.2B")

    def decide(self, snapshot, *, digest, timeout=20):
        self.snapshots.append(snapshot)
        self.release.wait(2)
        return dict(snapshot_id=snapshot["snapshot_id"], action="WAIT", reason="Нет данных", latency_ms=1)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.now = utc_now()
        self.config = Config(source="kraken-public", strategy_version="qwen-local-v1", llm_model=NINE,
                             llm_digest=DIGEST)
        self.state = State("nine", stamp(self.now), stamp(self.now+timedelta(hours=1)), "RUNNING", "1000", "1000")
        self.quote = Quote("q", stamp(self.now), D("100"), D("100.01"), self.config.source)
        self.snapshot = build_snapshot(self.config, self.state, self.quote, self.now)

    def test_config_and_panel_accept_only_two_models_with_compatible_default(self):
        config, _ = parse_settings(dict(mode="paper", strategy="qwen", llm_model=NINE))
        self.assertEqual(config.llm_model, NINE)
        self.assertEqual(Config.from_dict(config.to_dict()).llm_model, NINE)
        old = config.to_dict(); old.pop("llm_model")
        self.assertEqual(Config.from_dict(old).llm_model, MODEL)
        for tag in ("tgai:latest", "qwen3.5:9b-cloud", "http://remote/model", "other", None):
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    replace(config, llm_model=tag)
                with self.assertRaises(ValueError):
                    OllamaClient(model=tag)

    def test_nine_transport_checks_name_size_digest_and_snapshot(self):
        client = OllamaClient(model=NINE)
        row = dict(name=NINE, digest=DIGEST, details=dict(family="qwen35", parameter_size="8.95B"))
        with patch.object(client, "_request", return_value=dict(models=[row])):
            self.assertEqual(client.status()["model"], NINE)
            for details in (dict(family="other", parameter_size="8.95B"),
                            dict(family="qwen35", parameter_size="4.2B")):
                with patch.object(client, "_request", return_value=dict(models=[dict(row, details=details)])):
                    with self.assertRaises(ModelError):
                        client.status()
        choice = self.snapshot["decision_contract"][0]
        raw = dict(model=NINE, done=True, message=dict(content=json.dumps(dict(
            snapshot_id=self.snapshot["snapshot_id"], **choice))))
        with patch.object(client, "status", return_value=dict(digest=DIGEST)), \
                patch.object(client, "_request", return_value=raw) as request:
            answer = client.decide(self.snapshot, digest=DIGEST)
            self.assertEqual(request.call_args.args[1]["model"], NINE)
            self.assertLessEqual(request.call_args.kwargs["timeout"], 25)
            self.assertGreater(request.call_args.kwargs["timeout"], 24)
            self.assertEqual(answer["explanation_status"], "VERIFIED")
        with patch.object(client, "status", return_value=dict(digest="a"*64)):
            with self.assertRaises(ModelError):
                client.decide(self.snapshot, digest=DIGEST)
        with patch.object(client, "status") as status:
            with self.assertRaises(ModelError):
                client.decide(dict(self.snapshot, model=MODEL), digest=DIGEST)
            status.assert_not_called()
        with patch.object(client, "status", return_value=dict(digest=DIGEST)), \
                patch.object(client, "_request", return_value=dict(raw, model=MODEL)):
            with self.assertRaises(ModelError):
                client.decide(self.snapshot, digest=DIGEST)

    def test_snapshot_hash_binds_model_and_four_keeps_old_shape(self):
        four = build_snapshot(replace(self.config, llm_model=MODEL), self.state, self.quote, self.now)
        self.assertEqual(self.snapshot["model"], NINE)
        self.assertNotIn("model", four)
        self.assertNotEqual(self.snapshot["snapshot_id"], four["snapshot_id"])

    def test_connection_checks_selected_model_and_clears_previous_success_on_failure(self):
        clients = []
        def factory(model):
            client = ProbeModel(model)
            client.fail = model == NINE
            clients.append(client)
            return client
        connection = ModelConnection(client_factory=factory)
        connection.connect(MODEL)
        await_condition(lambda:not connection.view()["checking"])
        self.assertTrue(connection.view()["connected"])
        connection.connect(NINE)
        await_condition(lambda:not connection.view()["checking"])
        view = connection.view()
        self.assertFalse(view["connected"])
        self.assertEqual(view["model"], NINE)
        self.assertIsNone(view["info"])
        self.assertIsNone(view["last_answer"])
        self.assertEqual(clients[0].snapshots[0]["allowed_actions"], ["WAIT"])

    def test_connection_cannot_switch_during_probe_and_rejects_wrong_model_info(self):
        client = ProbeModel(NINE); client.release.clear()
        connection = ModelConnection(client_factory=lambda model:client)
        try:
            connection.connect(NINE)
            await_condition(lambda:bool(client.snapshots))
            with self.assertRaises(ModelError):
                connection.connect(MODEL)
            self.assertEqual(connection.connect(NINE)["model"], NINE)
        finally:
            client.release.set()
        await_condition(lambda:not connection.view()["checking"])
        self.assertTrue(connection.view()["connected"])
        wrong = ModelConnection(client_factory=lambda model:ProbeModel(MODEL))
        wrong.connect(NINE)
        await_condition(lambda:not wrong.view()["checking"])
        self.assertFalse(wrong.view()["connected"])

    def test_new_session_pins_connected_model_and_restore_uses_same_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sessions.sqlite3"
            manager = SessionManager(path)
            manager.model = ModelConnection(client_factory=ProbeModel)
            manager.model.connect(MODEL)
            await_condition(lambda:not manager.model.view()["checking"])
            with self.assertRaises(ValueError):
                manager.launch(payload=dict(mode="paper", strategy="qwen", llm_model=NINE))
            manager.model.connect(NINE)
            await_condition(lambda:not manager.model.view()["checking"])
            def worker(config, minutes, session_id, ready, startup):
                store = Store(path)
                try:
                    engine = Engine.start(store, config, self.now, self.now+timedelta(minutes=minutes))
                    startup["id"] = engine.state.session_id
                    engine.strategy.close()
                finally:
                    store.close()
                ready.set()
            with patch.object(manager, "_worker", side_effect=worker):
                session_id = manager.launch(payload=dict(mode="paper", strategy="qwen", llm_model=NINE))
            manager.thread.join()
            store = Store(path)
            try:
                saved, _ = store.load(session_id)
                self.assertEqual(saved.llm_model, NINE)
                self.assertEqual(saved.llm_digest, DIGEST)
                engine = Engine(store, session_id)
                self.assertEqual(engine.strategy.client.model, NINE)
                engine.strategy.close()
            finally:
                store.close()

    def test_manager_refuses_model_change_while_session_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(Path(directory)/"sessions.sqlite3")
            with patch.object(SessionManager, "running", new_callable=unittest.mock.PropertyMock, return_value=True), \
                    patch.object(manager.model, "connect") as connect:
                with self.assertRaises(ValueError):
                    manager.model_change(dict(model=NINE))
                connect.assert_not_called()

    def test_nine_replay_rejects_model_mismatch_in_config_request_or_answer(self):
        export = dataset(qwen=True)
        export["config"]["llm_model"] = NINE
        request, answer = add_decision(export)
        request["payload"]["model"] = answer["payload"]["model"] = NINE
        self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"], 1)
        for place in ("config", "request", "answer"):
            changed = deepcopy(export)
            if place == "config":
                changed["config"]["llm_model"] = MODEL
            else:
                event = next(e for e in changed["events"] if e["kind"] == ("AI_REQUEST" if place == "request" else "AI_DECISION"))
                event["payload"]["model"] = MODEL
            self.assertEqual(compare_session(changed)["qwen_replay"]["replayed_decisions"], 0)

    def test_panel_http_connects_selected_model_and_keeps_csrf_and_start_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(Path(directory)/"sessions.sqlite3")
            manager.model = ModelConnection(client_factory=ProbeModel)
            server = DashboardServer(("127.0.0.1", 0), manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            def request(method, path, payload=None, token=None):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                headers = {"Content-Type":"application/json"}
                if token:
                    headers["X-Control-Token"] = token
                conn.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=headers)
                reply = conn.getresponse(); result = reply.status, json.loads(reply.read()); conn.close()
                return result
            try:
                status, view = request("GET", "/api/state")
                self.assertIn("qwen-model-selection-v2", view["capabilities"])
                token = view["control_token"]
                self.assertEqual(request("POST", "/api/model/connect", dict(model=NINE))[0], 403)
                self.assertEqual(request("POST", "/api/model/connect", dict(model=NINE), token)[0], 200)
                await_condition(lambda:not manager.model.view()["checking"])
                status, model = request("GET", "/api/model/state")
                self.assertTrue(model["connected"])
                self.assertEqual(model["model"], NINE)
                self.assertEqual(request("POST", "/api/model/connect", dict(model="tgai:latest"), token)[0], 400)
                self.assertEqual(request("POST", "/api/start", dict(mode="paper", strategy="qwen", llm_model=MODEL), token)[0], 400)
                self.assertFalse(manager.running)
            finally:
                server.shutdown(); server.server_close(); thread.join(); manager.shutdown()


if __name__ == "__main__":
    unittest.main()
