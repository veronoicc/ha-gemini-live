"""Tests for the Gemini Live adapter's barge-in behaviour."""

from types import SimpleNamespace

from gemini_live.gemini import (
    GeminiLiveClient,
    GeminiLiveSession,
    _escape_decode,
    _gemini_config,
    _gemini_schema,
    _gemini_tool,
)
from gemini_live.live import LiveConfig, LiveEvent, LiveTool, LiveToolResponse


def _make_config(**overrides) -> LiveConfig:
    defaults = {"model": "m", "voice": "v", "system_instruction": "s"}
    defaults.update(overrides)
    return LiveConfig(**defaults)


def _server_content(**overrides):
    defaults = {
        "model_turn": None,
        "output_transcription": None,
        "input_transcription": None,
        "interrupted": False,
        "turn_complete": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _sdk_response(server_content=None, **attributes):
    return SimpleNamespace(
        tool_call=None,
        server_content=server_content,
        go_away=None,
        session_resumption_update=None,
        **attributes,
    )


class _FakeSDKSession:
    def __init__(self, responses):
        self._responses = responses

    async def receive(self):
        for response in self._responses:
            yield response


class _TurnBoundedFakeSDKSession:
    """Model the SDK returning one iterator per completed server turn."""

    def __init__(self, turns):
        self._turns = iter(turns)
        self.receive_count = 0

    async def receive(self):
        self.receive_count += 1
        for response in next(self._turns):
            yield response


class _FakeConnectContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_exc):
        return None


class _FakeSDKClient:
    def __init__(self, session):
        self.context = _FakeConnectContext(session)
        self.aio = SimpleNamespace(
            live=SimpleNamespace(connect=lambda **_kwargs: self.context)
        )


async def _collect(session: GeminiLiveSession) -> list[LiveEvent]:
    return [event async for event in session.receive()]


def test_gemini_config_legacy_keeps_existing_realtime_input_config():
    config = _gemini_config(_make_config())

    assert config["realtime_input_config"] == {
        "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY",
        "activity_handling": "NO_INTERRUPTION",
    }

def test_gemini_config_barge_in_enables_activity_interruption():
    config = _gemini_config(_make_config(support_barge_in=True))

    realtime_input = config["realtime_input_config"]
    assert realtime_input["automatic_activity_detection"]["disabled"] is False
    assert realtime_input["activity_handling"] == "START_OF_ACTIVITY_INTERRUPTS"
    assert realtime_input["turn_coverage"] == "TURN_INCLUDES_ONLY_ACTIVITY"


def test_gemini_config_barge_in_does_not_change_other_settings():
    legacy = _gemini_config(_make_config())
    barge_in = _gemini_config(_make_config(support_barge_in=True))

    for key in legacy:
        if key != "realtime_input_config":
            assert legacy[key] == barge_in[key]


async def test_gemini_interrupted_event_is_normalized():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(interrupted=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=False)]


async def test_gemini_combined_interrupted_and_turn_complete_single_event():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(interrupted=True, turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=True)]


async def test_gemini_plain_turn_complete_remains_terminal_marker():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert events == [LiveEvent(turn_complete=True)]


async def test_gemini_server_content_without_interrupted_attribute():
    # Older SDK payloads may not carry the interrupted attribute at all.
    content = SimpleNamespace(
        model_turn=None,
        output_transcription=None,
        input_transcription=None,
        turn_complete=True,
    )
    session = GeminiLiveSession(_FakeSDKSession([_sdk_response(content)]))

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=False, turn_complete=True)]


async def test_gemini_full_interrupted_sequence_normalization():
    session = GeminiLiveSession(
        _FakeSDKSession(
            [
                _sdk_response(_server_content(turn_complete=True, interrupted=False)),
                _sdk_response(_server_content(interrupted=True, turn_complete=True)),
            ]
        )
    )

    events = await _collect(session)

    assert [event.turn_complete for event in events] == [True, True]
    assert [event.interrupted for event in events] == [False, True]


async def test_gemini_barge_in_reenters_sdk_receive_for_replacement_turn():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [
                _sdk_response(_server_content(interrupted=True)),
                _sdk_response(
                    _server_content(turn_complete=True)
                )
            ],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    session = GeminiLiveSession(sdk_session, support_barge_in=True)

    events = await _collect(session)

    assert events == [
        LiveEvent(interrupted=True),
        LiveEvent(turn_complete=True),
        LiveEvent(turn_complete=True),
    ]
    assert sdk_session.receive_count == 2


async def test_gemini_client_passes_barge_in_mode_to_session():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [_sdk_response(_server_content(interrupted=True, turn_complete=True))],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    client = GeminiLiveClient(_FakeSDKClient(sdk_session))

    async with client.connect(_make_config(support_barge_in=True)) as session:
        events = await _collect(session)

    assert events[-1] == LiveEvent(turn_complete=True)
    assert sdk_session.receive_count == 2


async def test_gemini_legacy_does_not_reenter_sdk_receive():
    sdk_session = _TurnBoundedFakeSDKSession(
        [
            [
                _sdk_response(
                    _server_content(interrupted=True, turn_complete=True)
                )
            ],
            [_sdk_response(_server_content(turn_complete=True))],
        ]
    )
    session = GeminiLiveSession(sdk_session, support_barge_in=False)

    events = await _collect(session)

    assert events == [LiveEvent(interrupted=True, turn_complete=True)]
    assert sdk_session.receive_count == 1


def test_gemini_schema_handles_anyof_and_oneof():
    anyof_schema = {
        "anyOf": [
            {"type": "string", "description": "Single entity ID"},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    converted = _gemini_schema(anyof_schema)
    assert converted["type"] == "STRING"
    assert converted["description"] == "Single entity ID"

    oneof_schema = {
        "oneOf": [
            {"type": "integer"},
            {"type": "number"},
        ]
    }
    assert _gemini_schema(oneof_schema)["type"] == "INTEGER"


def test_gemini_schema_removes_required_from_non_object_types():
    schema = {
        "type": "string",
        "required": ["area_id"],
        "properties": {"foo": {"type": "string"}},
    }
    converted = _gemini_schema(schema)
    assert converted["type"] == "STRING"
    assert "required" not in converted
    assert "properties" not in converted


def test_gemini_schema_filters_required_to_existing_properties():
    schema = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target", "nonexistent"],
    }
    converted = _gemini_schema(schema)
    assert converted["type"] == "OBJECT"
    assert converted["required"] == ["target"]


def test_gemini_tool_ensures_top_level_object():
    tool = LiveTool(name="test_primitive", description="Desc", parameters={"type": "string"})
    declaration = _gemini_tool(tool)
    assert declaration["parameters"]["type"] == "OBJECT"
    assert "properties" in declaration["parameters"]

def test_gemini_tool_sets_blocking_behavior():
    tool = LiveTool(name="test_tool", description="A test tool", parameters={"type": "object"})
    declaration = _gemini_tool(tool)
    assert declaration["name"] == "test_tool"
    assert declaration["description"] == "A test tool"
    assert declaration["behavior"] == "BLOCKING"


def test_escape_decode_safe_with_stray_backslashes():
    # Backslashes not forming valid escapes must not crash
    assert _escape_decode(r"\xinvalid") == r"\xinvalid"
    assert _escape_decode(r"Room \ Kitchen") == r"Room \ Kitchen"
    assert _escape_decode({"key": r"\xinvalid"}) == {"key": r"\xinvalid"}
    assert _escape_decode([r"\unothex"]) == [r"\unothex"]

async def test_send_text_uses_send_client_content():
    calls = []

    class _FakeSessionWithContent:
        async def send_client_content(self, *, turns, turn_complete):
            calls.append((turns, turn_complete))

    session = GeminiLiveSession(_FakeSessionWithContent())
    await session.send_text("Hello Gemini")

    assert len(calls) == 1
    turns, turn_complete = calls[0]
    assert turn_complete is True
    assert len(turns) == 1
    assert turns[0].parts[0].text == "Hello Gemini"


async def test_send_tool_responses_wraps_primitive_results():
    calls = []

    class _FakeSessionWithToolResp:
        async def send_tool_response(self, *, function_responses):
            calls.append(function_responses)

    session = GeminiLiveSession(_FakeSessionWithToolResp())
    await session.send_tool_responses([
        LiveToolResponse(name="light_toggle", call_id="c1", response="Done"),
        LiveToolResponse(name="get_state", call_id="c2", response={"state": "on"}),
    ])

    assert len(calls) == 1
    responses = calls[0]
    assert len(responses) == 2
    assert responses[0].response == {"result": "Done"}
    assert responses[1].response == {"state": "on"}
