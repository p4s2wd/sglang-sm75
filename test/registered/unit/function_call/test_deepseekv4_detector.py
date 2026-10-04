"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import logging
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_streaming_waits_for_complete_invoke(self):
        detector = DeepSeekV4Detector()
        partial = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual(partial.calls, [])

        complete = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>",
            self.tools,
        )
        self.assertEqual(len(complete.calls), 1)
        self.assertEqual(complete.calls[0].name, "get_weather")
        self.assertEqual(complete.calls[0].parameters, '{"city": "{\\"a\\""}')
        self.assertEqual(
            detector.prev_tool_call_arr,
            [{"name": "get_weather", "arguments": complete.calls[0].parameters}],
        )
        self.assertEqual(
            detector.streamed_args_for_tool, [complete.calls[0].parameters]
        )

    def test_streaming_matches_complete_parse_at_every_chunk_width(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_tasks",
                    description="Create tasks",
                    parameters={
                        "type": "object",
                        "properties": {"tasks": {"type": "array"}},
                    },
                ),
            )
        ]
        values = (
            '[{"description":"first","priority":"medium"}]',
            '[{"description":"unterminated,"priority":"medium"}]',
            "[{]",
            "[}",
        )

        for value in values:
            text = _wrapped(_invoke("create_tasks", _param("tasks", "false", value)))
            expected = DeepSeekV4Detector().detect_and_parse(text, tools).calls[0]
            for width in range(1, len(text) + 1):
                with self.subTest(value=value, width=width):
                    chunks = [
                        text[index : index + width]
                        for index in range(0, len(text), width)
                    ]
                    normal, calls = self._feed_with_tools(chunks, tools)

                    self.assertEqual(normal.strip(), "")
                    self.assertNotIn(DSML, normal)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, "create_tasks")
                    self.assertEqual(calls[0].parameters, expected.parameters)

    def test_complete_invoke_emits_name_and_arguments_together(self):
        text = _weather_call()

        _, calls = self._feed([text[:-1], text[-1:]])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, '{"city": "SF"}')

    def test_multiple_invokes_emit_one_complete_item_each(self):
        text = _wrapped(
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        _, calls = self._feed([text])

        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in calls],
            [
                (0, "get_weather", '{"city": "SF"}'),
                (1, "get_weather", '{"city": "NY"}'),
            ],
        )

    def test_closed_invoke_rejects_incomplete_parameter(self):
        text = _wrapped(
            _invoke(
                "get_weather",
                f'<{DSML}parameter name="city" string="true">SF',
            )
        )

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_stray_prose_inside_invoke_is_ignored(self):
        """Prose around the parameter tags cannot hide a lost argument, so the
        call is still emitted, matching the one-shot parser and the previous
        streaming behavior."""
        text = _wrapped(
            _invoke(
                "get_weather",
                "thinking aloud\n" + _param("city", "true", "SF") + "\nnote to self",
            )
        )

        for width in (4, len(text)):
            with self.subTest(width=width):
                normal, calls = self._feed(
                    [text[i : i + width] for i in range(0, len(text), width)]
                )
                self.assertEqual(normal, "")
                self.assertEqual(
                    [(call.name, call.parameters) for call in calls],
                    [("get_weather", '{"city": "SF"}')],
                )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "SF"}')],
        )

    def test_closed_invoke_rejects_unmatched_parameter_tag(self):
        """Leftover text containing a DSML fragment is a parameter that failed
        to match; the call is dropped rather than emitted without it."""
        bodies = {
            "self-closing": f'<{DSML}parameter name="city" string="true"/>',
            "unclosed after valid": _param("city", "true", "SF")
            + f'<{DSML}parameter name="units" string="true">C',
        }
        for label, body in bodies.items():
            with self.subTest(label=label):
                normal, calls = self._feed([_wrapped(_invoke("get_weather", body))])
                self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_body_without_parameters(self):
        text = _wrapped(_invoke("get_weather", "just prose, no parameters"))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_malformed_direct_json(self):
        text = _wrapped(_invoke("get_weather", '{"city": }'))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_nonstandard_json_constant(self):
        text = _wrapped(_invoke("get_weather", '{"city": NaN}'))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_nonstandard_xml_constant_falls_back_to_string(self):
        text = _wrapped(_invoke("get_weather", _param("city", "false", "NaN")))

        _, calls = self._feed([text])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].parameters, '{"city": "NaN"}')

    def test_malformed_first_invoke_does_not_create_flush_state(self):
        detector = DeepSeekV4Detector()

        result = detector.parse_streaming_increment(
            _wrapped(_invoke("get_weather", '{"city": }')), self.tools
        )

        self.assertEqual(result.calls, [])
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.streamed_args_for_tool, [])

    def test_malformed_later_invoke_preserves_completed_call(self):
        text = _wrapped(
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n"
            + _invoke("get_weather", '{"city": }')
        )

        normal, calls = self._feed([text])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, '{"city": "SF"}')
        self.assertEqual(normal, "")

    def test_malformed_first_invoke_keeps_prose_and_later_call(self):
        """A malformed invoke is discarded on its own: the prose ahead of the
        block survives, the next well-formed invoke still becomes call 0, and
        no DSML reaches the client."""
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", f'<{DSML}parameter name="city">SF')
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        normal, calls = self._feed([text])

        self.assertEqual(normal, "Checking.")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in calls],
            [(0, "get_weather", '{"city": "NY"}')],
        )

    def test_malformed_invoke_fails_closed_at_every_chunk_width(self):
        """Whatever the delta boundaries, a malformed invoke yields the prose,
        no call, and no DSML, including the trailing tool_calls close."""
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", f'<{DSML}parameter name="city" string="true">SF')
        )

        for width in range(1, len(text) + 1):
            with self.subTest(width=width):
                parser = FunctionCallParser(self.tools, "deepseekv4")
                normal, calls = "", []
                for index in range(0, len(text), width):
                    chunk_normal, chunk_calls = parser.parse_stream_chunk(
                        text[index : index + width]
                    )
                    normal += chunk_normal
                    calls.extend(chunk_calls)
                end_normal, end_calls = parser.parse_stream_end()
                normal += end_normal
                calls.extend(end_calls)

                self.assertEqual(normal, "Checking.")
                self.assertEqual(calls, [])

    def test_non_streaming_drops_only_the_malformed_invoke(self):
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", '{"city": }')
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "NY"}')],
        )

    def test_non_streaming_drops_unclosed_invoke_in_closed_wrapper(self):
        text = (
            f"<{DSML}tool_calls>\n"
            f'<{DSML}invoke name="get_weather">\n{{"city": "SF"}}\n'
            f"</{DSML}tool_calls>"
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_non_streaming_parses_bare_invoke_without_wrapper(self):
        result = DeepSeekV4Detector().detect_and_parse(
            "Checking.\n\n" + _invoke("get_weather", '{"city": "SF"}'), self.tools
        )

        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "SF"}')],
        )

    def test_non_streaming_malformed_bare_invoke_fails_closed(self):
        result = DeepSeekV4Detector().detect_and_parse(
            _invoke("get_weather", '{"city": }'), self.tools
        )

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_stream_end_drops_incomplete_invoke(self):
        parser = FunctionCallParser(self.tools, "deepseekv4")
        normal, calls = parser.parse_stream_chunk(
            "Checking.\n\n"
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">SF'
        )

        self.assertEqual((normal, calls), ("", []))
        self.assertEqual(parser.parse_stream_end(), ("Checking.", []))

    def _feed_with_tools(self, chunks, tools):
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_unexpected_parse_error_fails_closed(self):
        """An unexpected parse error keeps the prose, drops the DSML and the
        call, and the dropped buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(
                "Checking.\n\n" + _weather_call(), self.tools
            )
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertEqual(first.normal_text, "Checking.")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.normal_text, " tail")


def _close(tag: str) -> str:
    """Closing tag for ``tag``, assembled so the spelling stays in one place."""
    return "</" + tag + ">"


class TestDeepSeekV4DriftTolerance(unittest.TestCase):
    """Drift observed in a long agentic session on 2026-10-04.

    A client that refuses a tool call over one capital letter turns a soft model
    slip into a retry loop, so a name that still identifies exactly one declared
    tool is repaired rather than rejected. A name that matches no declared tool
    is forwarded unchanged and warned about, leaving the client free to tell the
    model what it called and the log free to show what drifted.
    """

    def setUp(self):
        def declared(name, prop):
            return Tool(
                type="function",
                function=Function(
                    name=name,
                    description=f"{name} tool",
                    parameters={
                        "type": "object",
                        "properties": {prop: {"type": "string"}},
                        "required": [prop],
                    },
                ),
            )

        self.tools = [declared("bash", "command"), declared("read", "path")]

    @staticmethod
    def _param(body, value, prefixed_closer=True):
        closer = _close(DSML + "parameter") if prefixed_closer else _close("parameter")
        return f"<{DSML}parameter {body}>{value}{closer}"

    def _invoke(self, name, parameter):
        return (
            f"<{DSML}tool_calls>\n"
            f'<{DSML}invoke name="{name}">\n{parameter}\n'
            f"{_close(DSML + 'invoke')}\n{_close(DSML + 'tool_calls')}"
        )

    def _stream(self, text, width=7):
        """Parse in small deltas: the repair must not depend on chunk width."""
        detector = DeepSeekV4Detector()
        calls = []
        for start in range(0, len(text), width):
            result = detector.parse_streaming_increment(
                text[start : start + width], self.tools
            )
            calls.extend(result.calls)
        calls.extend(detector.parse_streaming_increment("", self.tools).calls)
        return [(call.name, call.parameters) for call in calls]

    def test_parameter_without_string_attribute_is_still_parsed(self):
        """A missing string attribute used to cost the whole call; the value is
        recovered as a string instead."""
        text = self._invoke("bash", self._param('name="command"', "echo hi"))

        self.assertEqual(self._stream(text), [("bash", '{"command": "echo hi"}')])

    def test_loose_string_attribute_is_still_parsed(self):
        bodies = {
            "unquoted": 'name="command" string=true',
            "capital value": 'name="command" string="True"',
            "padded": 'name="command" string = "true"',
        }
        for label, body in bodies.items():
            with self.subTest(label=label):
                text = self._invoke("bash", self._param(body, "pwd"))

                self.assertEqual(self._stream(text), [("bash", '{"command": "pwd"}')])

    def test_closer_without_dsml_prefix_is_accepted(self):
        """The exact production shape: a correct parameter closed as a plain
        html tag. Dropping it stalled the agent loop."""
        text = self._invoke(
            "bash",
            self._param(
                'name="command" string="true"', "echo hello", prefixed_closer=False
            ),
        )

        self.assertEqual(self._stream(text), [("bash", '{"command": "echo hello"}')])

    def test_tool_name_drift_is_repaired_to_the_declared_name(self):
        for spelled, declared in [
            ("Bash", "bash"),
            ("BASH", "bash"),
            (" bash ", "bash"),
            ("functions.bash", "bash"),
            ("Read", "read"),
        ]:
            with self.subTest(spelled=spelled):
                text = self._invoke(
                    spelled, self._param('name="command" string="true"', "ls")
                )

                self.assertEqual(self._stream(text), [(declared, '{"command": "ls"}')])
                one_shot = DeepSeekV4Detector().detect_and_parse(text, self.tools)
                self.assertEqual(
                    [(call.name, call.parameters) for call in one_shot.calls],
                    [(declared, '{"command": "ls"}')],
                )

    def test_quoted_name_is_repaired_where_names_arrive_as_json(self):
        """The direct-JSON shape and every other detector hand the parsed name
        straight to parse_base_json, so the quote repair belongs there."""
        detector = DeepSeekV4Detector()
        for spelled, declared in [
            ('"bash"', "bash"),
            (" bash ", "bash"),
            ("BASH", "bash"),
            ("functions.read", "read"),
        ]:
            with self.subTest(spelled=spelled):
                calls = detector.parse_base_json(
                    {"name": spelled, "parameters": {}}, self.tools
                )

                self.assertEqual([call.name for call in calls], [declared])

    def test_unknown_tool_name_is_forwarded_with_a_warning(self):
        text = self._invoke(
            "exec_command", self._param('name="cmd" string="true"', "ls -la")
        )

        with self.assertLogs(
            "sglang.srt.function_call.deepseekv32_detector", logging.WARNING
        ) as logged:
            calls = self._stream(text)

        self.assertEqual(calls, [("exec_command", '{"cmd": "ls -la"}')])
        self.assertTrue(
            any("not in the tool list" in r.getMessage() for r in logged.records),
            "an unrecognised name must leave a trace in the server log",
        )

    def test_unparseable_parameter_still_fails_closed(self):
        """Tolerance is bounded: a parameter that never closes must not leak a
        half-read argument into the call."""
        text = self._invoke(
            "bash",
            self._param('name="command" string="true"', "ls")
            + f'<{DSML}parameter name="timeout" string="true">5',
        )

        self.assertEqual(self._stream(text), [])


if __name__ == "__main__":
    unittest.main()
