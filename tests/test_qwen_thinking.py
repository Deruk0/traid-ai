"""Bounded thinking preserves the final contract and archived session settings."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import unittest
from unittest.mock import patch

from traid_ai.evaluation import compare_session
from traid_ai.models import Config, D, Quote, State, stamp, utc_now
from traid_ai.qwen import MODEL, ModelError, OllamaClient, build_snapshot
from test_evaluation import dataset, add_decision


class ThinkingTests(unittest.TestCase):
    def setUp(self):
        self.now = utc_now()
        self.config = Config(source="kraken-public", strategy_version="qwen-local-v1", llm_digest="a"*64)
        self.state = State("thinking-test", stamp(self.now), stamp(self.now+timedelta(hours=1)),
                           "RUNNING", "1000", "1000")
        self.quote = Quote("quote", stamp(self.now), D("100"), D("100.01"), self.config.source)
        self.snapshot = build_snapshot(self.config, self.state, self.quote, self.now)
        choice = next(c for c in self.snapshot["decision_contract"] if c["action"] == "WAIT")
        self.wire = dict(snapshot_id=self.snapshot["snapshot_id"], **choice)
        self.client = OllamaClient()
        self.status = patch.object(self.client, "status", return_value=dict(digest=self.config.llm_digest))
        self.status.start()
        self.addCleanup(self.status.stop)

    def reply(self, *, reason="stop", thinking="", content=None, model=MODEL):
        return dict(model=model, done=True, done_reason=reason,
                    message=dict(thinking=thinking, content=json.dumps(self.wire) if content is None else content),
                    prompt_eval_count=100, eval_count=50)

    def decide(self):
        return self.client.decide(self.snapshot, digest=self.config.llm_digest)

    def test_thinking_returns_same_verified_json_without_exposing_analysis(self):
        with patch.object(self.client, "_request", return_value=self.reply(thinking="private-analysis")) as request:
            answer = self.decide()
        payload = request.call_args.args[1]
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertNotIn("tools", payload)
        self.assertTrue(all(set(branch["required"]) == set(self.wire)
                            for branch in payload["format"]["oneOf"]))
        self.assertEqual(answer["action"], "WAIT")
        self.assertEqual(answer["explanation_status"], "VERIFIED")
        self.assertNotIn("private-analysis", json.dumps(answer))
        self.assertEqual(answer["inference_stages"], 1)

    def test_capped_analysis_finalizes_with_remaining_budget_and_original_schema(self):
        draft = "private-draft: REQUEST_ENTRY is only a hypothesis"
        first = self.reply(reason="length", thinking=draft, content="")
        second = self.reply()
        with patch.object(self.client, "_request", side_effect=[first, second]) as request, \
                patch("traid_ai.qwen.time.monotonic", side_effect=[10, 10, 18, 19]):
            answer = self.decide()
        initial, final = request.call_args_list
        self.assertTrue(initial.args[1]["think"])
        self.assertFalse(final.args[1]["think"])
        self.assertEqual(final.kwargs["timeout"], 12)
        self.assertEqual(final.args[1]["options"]["num_predict"], 256)
        self.assertEqual(final.args[1]["format"], initial.args[1]["format"])
        self.assertIn(draft, final.args[1]["messages"][-2]["content"])
        self.assertEqual(answer["action"], "WAIT")
        self.assertEqual(answer["inference_stages"], 2)
        self.assertTrue(answer["thinking_capped"])
        self.assertEqual(answer["output_tokens"], 100)
        self.assertEqual(answer["prompt_tokens"], 200)
        self.assertEqual(answer["latency_ms"], 9000)
        self.assertNotIn(draft, json.dumps(answer))

    def test_expired_budget_never_starts_finalization(self):
        with patch.object(self.client, "_request", return_value=self.reply(reason="length", thinking="draft")) as request, \
                patch("traid_ai.qwen.time.monotonic", side_effect=[10, 10, 31]):
            with self.assertRaises(ModelError):
                self.decide()
        self.assertEqual(request.call_count, 1)

    def test_invalid_or_missing_analysis_never_starts_second_request(self):
        for raw in (self.reply(reason="length", thinking="", content="{}"),
                    self.reply(reason="length", thinking="draft", model="other"),
                    self.reply(reason="length", thinking=123)):
            with self.subTest(raw=raw), patch.object(self.client, "_request", return_value=raw) as request:
                with self.assertRaises(ModelError):
                    self.decide()
                self.assertEqual(request.call_count, 1)

    def test_incomplete_invalid_or_fabricated_final_answer_is_rejected(self):
        first = self.reply(reason="length", thinking="draft", content="")
        invalid = dict(self.wire, evidence_refs=["made-up-fact"])
        for final in (self.reply(reason="length"), self.reply(content="not-json"),
                      self.reply(content=json.dumps(invalid))):
            with self.subTest(final=final), patch.object(self.client, "_request", side_effect=[first, final]):
                with self.assertRaises(ModelError):
                    self.decide()

    def test_finalization_transport_failure_is_not_retried(self):
        first = self.reply(reason="length", thinking="draft", content="")
        with patch.object(self.client, "_request", side_effect=[first, ModelError("timeout")]) as request:
            with self.assertRaises(ModelError):
                self.decide()
        self.assertEqual(request.call_count, 2)

    def test_old_config_restores_original_direct_inference(self):
        payload = self.config.to_dict()
        payload.pop("llm_thinking")
        restored = Config.from_dict(payload)
        self.assertFalse(restored.llm_thinking)
        self.snapshot = build_snapshot(restored, self.state, self.quote, self.now)
        self.assertNotIn("inference", self.snapshot)
        self.wire["snapshot_id"] = self.snapshot["snapshot_id"]
        with patch.object(self.client, "_request", return_value=self.reply()) as request:
            self.decide()
        self.assertEqual(request.call_count, 1)
        options = request.call_args.args[1]
        self.assertFalse(options["think"])
        self.assertEqual(options["options"], dict(temperature=0, num_predict=256, num_ctx=4096))
        self.assertTrue(Config.from_dict(self.config.to_dict()).llm_thinking)
        for value in ("true", 1, None):
            with self.assertRaises(ValueError):
                replace(self.config, llm_thinking=value)

    def test_inference_is_hashed_and_unknown_settings_are_rejected(self):
        direct = build_snapshot(replace(self.config, llm_thinking=False), self.state, self.quote, self.now)
        self.assertNotEqual(direct["snapshot_id"], self.snapshot["snapshot_id"])
        self.snapshot["inference"]["num_predict"] = 99999
        with patch.object(self.client, "_request") as request:
            with self.assertRaises(ModelError):
                self.decide()
        request.assert_not_called()

    def test_replay_rejects_inference_mode_change_but_accepts_old_archives(self):
        for thinking in (False, True):
            export = dataset(qwen=True)
            export["config"]["llm_thinking"] = thinking
            add_decision(export)
            self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"], 1)
            changed = deepcopy(export)
            changed["config"]["llm_thinking"] = not thinking
            self.assertEqual(compare_session(changed)["qwen_replay"]["replayed_decisions"], 0)
            if not thinking:
                export["config"].pop("llm_thinking")
                self.assertEqual(compare_session(export)["qwen_replay"]["replayed_decisions"], 1)


if __name__ == "__main__":
    unittest.main()
