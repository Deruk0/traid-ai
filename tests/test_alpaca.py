from datetime import timedelta
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from traid_ai.alpaca import AlpacaError, NoRedirects, PaperClient, PaperConnection, PAPER_URL
from traid_ai.dashboard import DashboardServer, SessionManager
from traid_ai.models import stamp, utc_now


ACCOUNT = dict(id="paper-account-123456", status="ACTIVE", equity="100000",
               cash="90000", buying_power="90000", secret="must-not-be-returned")


class Reply(io.BytesIO):
    pass


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = PaperClient("test-key", "test-secret")

    def response(self, value):
        return Reply(json.dumps(value).encode())

    def test_only_paper_get_and_data_hosts_receive_credentials(self):
        replies = [ACCOUNT, [], [], {"quotes": {"BTC/USD": {
            "bp": 100, "ap": 101, "t": stamp(utc_now())}}}]
        calls = []
        def open_request(request, **kwargs):
            calls.append(request)
            return self.response(replies.pop(0))
        with patch.object(self.client._opener, "open", side_effect=open_request):
            account = self.client.account()
            self.client.positions()
            self.client.orders()
            quote = self.client.quote()
        self.assertNotIn("secret", account)
        self.assertNotIn("id", account)
        self.assertEqual(account["account_suffix"], "123456")
        self.assertFalse(quote["stale"])
        self.assertTrue(all(r.get_method() == "GET" for r in calls))
        self.assertTrue(all(r.full_url.startswith(PAPER_URL + "/") for r in calls[:3]))
        self.assertTrue(calls[3].full_url.startswith("https://data.alpaca.markets/"))
        self.assertEqual(calls[0].get_header("Apca-api-secret-key"), "test-secret")

    def test_http_failure_does_not_echo_secret(self):
        error = HTTPError(PAPER_URL, 401, "Unauthorized", {},
                          io.BytesIO(json.dumps({"message":"test-key test-secret", "code":40110000}).encode()))
        with patch.object(self.client._opener, "open", side_effect=error):
            with self.assertRaises(AlpacaError) as raised:
                self.client.account()
        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("test-key", str(raised.exception))
        self.assertIn("Paper Only", str(raised.exception))

    def test_timeout_and_malformed_account_are_visible(self):
        with patch.object(self.client._opener, "open", side_effect=URLError("offline")):
            with self.assertRaisesRegex(AlpacaError, "Нет связи"):
                self.client.account()
        for raw in ({}, dict(ACCOUNT, equity="NaN"), []):
            with patch.object(self.client._opener, "open", return_value=self.response(raw)):
                with self.assertRaises(AlpacaError):
                    self.client.account()

    def test_redirects_and_header_injection_are_rejected(self):
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "", {},
                                                       "https://evil.example"))
        for key in ("", "key\nInjected: 1", None):
            with self.assertRaises(AlpacaError):
                PaperClient(key, "secret")

    def test_quote_age_and_invalid_market_values(self):
        raw = {"quotes": {"BTC/USD": {"bp": 100, "ap": 101,
                                     "t": stamp(utc_now() - timedelta(minutes=1))}}}
        with patch.object(self.client._opener, "open", return_value=self.response(raw)):
            self.assertTrue(self.client.quote()["stale"])
        for bp in ("NaN", 0, 102):
            raw["quotes"]["BTC/USD"]["bp"] = bp
            with patch.object(self.client._opener, "open", return_value=self.response(raw)):
                with self.assertRaises(AlpacaError):
                    self.client.quote()

    def test_writes_are_fixed_paper_orders_only(self):
        order = dict(symbol="BTC/USD", side="buy", qty="0.0001", type="limit",
                     time_in_force="ioc", limit_price="100", client_order_id="test-order")
        with patch.object(self.client._opener, "open", return_value=self.response({"id":"paper-order"})) as opened:
            self.client.submit_order(order)
            request = opened.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, PAPER_URL + "/v2/orders")
            self.assertEqual(json.loads(request.data), order)
        for invalid in (dict(order,symbol="AAPL"), dict(order,type="market"),
                        dict(order,qty="-1"), dict(order,evil=True)):
            with self.assertRaises(AlpacaError):
                self.client.submit_order(invalid)
        with patch.object(self.client._opener, "open", return_value=Reply(b"")) as opened:
            self.client.cancel_order("paper-order")
            self.assertEqual(opened.call_args.args[0].get_method(), "DELETE")
            self.assertTrue(opened.call_args.args[0].full_url.startswith(PAPER_URL))

    def test_network_read_retries_once_but_post_is_never_replayed(self):
        with patch.object(self.client._opener, "open", side_effect=[
                URLError("offline"), self.response(ACCOUNT)]) as opened:
            self.assertEqual(self.client.account()["status"], "ACTIVE")
            self.assertEqual(opened.call_count, 2)
        order = dict(symbol="BTC/USD", side="buy", qty="0.001", type="limit",
                     time_in_force="ioc", limit_price="100", client_order_id="unknown")
        with patch.object(self.client._opener, "open", side_effect=URLError("lost")) as opened:
            with self.assertRaises(AlpacaError):
                self.client.submit_order(order)
            self.assertEqual(opened.call_count, 1)


class FakeClient:
    def __init__(self, key, secret):
        if key != "test-key" or secret != "test-secret":
            raise AlpacaError("Неверные тестовые ключи")
        self.fail = False
    def account(self):
        return dict(equity="100000", cash="90000", buying_power="90000", account_suffix="123456",
                    status="ACTIVE", currency="USD", trading_blocked=False, account_blocked=False)
    def positions(self):
        if self.fail:
            raise AlpacaError("Нет связи")
        return [dict(symbol="BTC/USD", qty="0.001", market_value="100")]
    def orders(self):
        return [dict(symbol="BTC/USD", status="partially_filled", qty="0.002", filled_qty="0.001")]
    def quote(self):
        return dict(symbol="BTC/USD", bid="100", ask="101", observed_at=stamp(utc_now()), stale=False)


def await_condition(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Ожидаемое состояние не достигнуто")


class ConnectionTests(unittest.TestCase):
    def test_connection_poll_failure_and_disconnect(self):
        client = FakeClient("test-key", "test-secret")
        connection = PaperConnection(client_factory=lambda k, s: client, poll_seconds=0.02)
        try:
            connection.connect("test-key", "test-secret")
            await_condition(lambda: bool(connection.view()["positions"]))
            view = connection.view()
            self.assertEqual(view["orders"][0]["status"], "partially_filled")
            last_time = view["resource_times"]["positions"]
            client.fail = True
            await_condition(lambda: "positions" in connection.view()["errors"])
            self.assertEqual(connection.view()["resource_times"]["positions"], last_time)
            self.assertTrue(connection.view()["positions"])
            self.assertNotIn("test-secret", json.dumps(view))
            connection.disconnect()
            time.sleep(0.05)
            self.assertFalse(connection.view()["connected"])
            self.assertEqual(connection.view()["positions"], [])
        finally:
            connection.shutdown()

    def test_inflight_result_cannot_reconnect_after_disconnect(self):
        gate, entered = threading.Event(), threading.Event()
        client = FakeClient("test-key", "test-secret")
        def blocked_positions():
            entered.set()
            gate.wait(2)
            return [dict(symbol="BTC/USD")]
        client.positions = blocked_positions
        connection = PaperConnection(client_factory=lambda k, s: client)
        try:
            connection.connect("test-key", "test-secret")
            self.assertTrue(entered.wait(1))
            connection.disconnect()
            gate.set()
            time.sleep(0.05)
            self.assertFalse(connection.view()["connected"])
        finally:
            gate.set()
            connection.shutdown()

    def test_environment_keys_without_file_storage(self):
        with patch.dict("os.environ", {"ALPACA_API_KEY":"test-key",
                                      "ALPACA_SECRET_KEY":"test-secret"}):
            connection = PaperConnection(client_factory=FakeClient)
            try:
                self.assertTrue(connection.view()["environment_keys_available"])
                self.assertTrue(connection.connect()["connected"])
            finally:
                connection.shutdown()


class HttpTests(unittest.TestCase):
    def test_local_form_connect_state_disconnect_and_csrf(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(Path(directory) / "sessions.sqlite3")
            manager.alpaca = PaperConnection(client_factory=FakeClient)
            server = DashboardServer(("127.0.0.1", 0), manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def request(path, payload=None, token=True):
                c = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-Control-Token"] = server.token
                c.request("GET" if payload is None else "POST", path,
                          json.dumps(payload) if payload is not None else None, headers)
                response = c.getresponse()
                result = response.status, json.loads(response.read())
                c.close()
                return result
            try:
                payload = {"key":"test-key", "secret":"test-secret"}
                self.assertEqual(request("/api/alpaca/connect", payload, token=False)[0], 403)
                self.assertFalse(manager.alpaca.view()["connected"])
                status, data = request("/api/alpaca/connect", payload)
                self.assertEqual(status, 200)
                self.assertTrue(data["connected"])
                self.assertNotIn("test-secret", json.dumps(data))
                await_condition(lambda: bool(manager.alpaca.view()["orders"]))
                self.assertEqual(request("/api/alpaca/state")[1]["orders"][0]["filled_qty"], "0.001")
                self.assertEqual(request("/api/alpaca/orders", payload)[0], 404)
                self.assertEqual(request("/api/alpaca/disconnect", {})[0], 200)
                self.assertFalse(request("/api/alpaca/state")[1]["connected"])
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                manager.shutdown()
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
