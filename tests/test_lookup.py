import json
import urllib.error

import pytest

from live_translation.lookup import (
    LOOKUP_SCHEMA,
    OllamaWordLookup,
    WordLookupError,
    context_window,
    mark_word,
    parse_lookup_response,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")


def test_context_window_keeps_short_text_and_word_offsets():
    text = "Let me see if this is loaded, oh so it is"
    start = text.index("loaded")

    context, s, e = context_window(text, start, start + len("loaded"))

    assert context == text
    assert context[s:e] == "loaded"


def test_context_window_trims_long_text_with_ellipses():
    text = " ".join(f"w{i}" for i in range(100))
    start = text.index(" w50 ") + 1

    context, s, e = context_window(text, start, start + 3, words_before=2, words_after=1)

    assert context == "… w48 w49 w50 w51 …"
    assert context[s:e] == "w50"


def test_context_window_keeps_punctuation_glued_to_word():
    text = 'he said "redemption," and left'
    start = text.index("redemption")

    context, s, e = context_window(
        text, start, start + len("redemption"), words_before=0, words_after=0
    )

    assert mark_word(context, s, e) == '… "[[redemption]]," …'


def test_context_window_rejects_range_outside_text():
    with pytest.raises(ValueError):
        context_window("abc", 2, 10)


def test_parse_lookup_response_normalizes_fields():
    content = (
        '```json\n{"lemma": "To Miss", "phrase": " miss  something ", '
        '"gloss": "упускаю", "explanation": "Не замечаю\\nчего-то."}\n```'
    )

    assert parse_lookup_response(content, "missing") == {
        "lemma": "miss",
        "phrase": "miss something",
        "gloss": "упускаю",
        "explanation": "Не замечаю чего-то.",
    }


def test_parse_lookup_response_falls_back_to_clicked_word_for_lemma():
    content = json.dumps({"lemma": "", "phrase": "", "gloss": "обычно", "explanation": ""})

    assert parse_lookup_response(content, "Typically")["lemma"] == "typically"


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "[1, 2]",
        '{"lemma": "x", "gloss": ',
        json.dumps({"lemma": "x", "phrase": "", "gloss": "", "explanation": ""}),
    ],
)
def test_parse_lookup_response_rejects_unusable_answers(content):
    with pytest.raises(WordLookupError):
        parse_lookup_response(content, "x")


def test_lookup_sends_one_chat_request_with_schema_and_marked_word(monkeypatch):
    calls = []
    answer = {
        "lemma": "figure",
        "phrase": "figure out",
        "gloss": "понял",
        "explanation": "Разобрался.",
    }

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, json.loads(req.data.decode("utf-8"))))
        return FakeResponse({"message": {"content": json.dumps(answer, ensure_ascii=False)}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    text = "I finally figured out why the build kept failing."
    start = text.index("figured")
    lookup = OllamaWordLookup(model="qwen3.5:4b", url="http://127.0.0.1:11434/", language="ru")

    result = lookup.lookup(text, start, start + len("figured"))

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen3.5:4b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"] == LOOKUP_SCHEMA
    assert "Russian" in payload["messages"][0]["content"]
    assert "[[figured]] out" in payload["messages"][1]["content"]
    assert result == {
        "word": "figured",
        **answer,
        "context": text,
        "context_start": start,
        "context_end": start + len("figured"),
        "language": "ru",
        "model": "qwen3.5:4b",
    }


def test_lookup_reports_unreachable_ollama(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(WordLookupError, match="not responding"):
        OllamaWordLookup().lookup("hello world", 0, 5)
