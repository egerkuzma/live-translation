import json
import urllib.error

import pytest

from live_translation.ollama_client import (
    DEFAULT_OLLAMA_URL,
    check_ollama,
    is_local_url,
    resolve_ollama_url,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        (None, {}, DEFAULT_OLLAMA_URL),
        (None, {"OLLAMA_HOST": "  "}, DEFAULT_OLLAMA_URL),
        (None, {"OLLAMA_HOST": "192.168.1.50"}, "http://192.168.1.50:11434"),
        (None, {"OLLAMA_HOST": "my-server:11500"}, "http://my-server:11500"),
        (None, {"OLLAMA_HOST": "0.0.0.0"}, "http://127.0.0.1:11434"),
        (None, {"OLLAMA_HOST": "http://my-server"}, "http://my-server:80"),
        (None, {"OLLAMA_HOST": "https://ollama.example.com/"}, "https://ollama.example.com:443"),
        ("http://127.0.0.1:11434/", {"OLLAMA_HOST": "192.168.1.50"}, "http://127.0.0.1:11434"),
    ],
)
def test_resolve_ollama_url_follows_flag_then_env_then_default(flag, env, expected):
    assert resolve_ollama_url(flag, env) == expected


def test_is_local_url():
    assert is_local_url("http://127.0.0.1:11434")
    assert is_local_url("http://localhost:11434")
    assert not is_local_url("http://192.168.1.50:11434")


def fake_server(monkeypatch, tags, running):
    seen = []

    def fake_urlopen(req, timeout):
        seen.append((req.full_url, timeout))
        path = req.full_url.rsplit("/api/", 1)[1]
        body = {
            "version": {"version": "0.34.0"},
            "tags": {"models": tags},
            "ps": {"models": running},
        }[path]
        return FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def test_check_ollama_reports_pulled_and_loaded_models(monkeypatch):
    seen = fake_server(
        monkeypatch,
        tags=[{"name": "qwen3.5:4b", "model": "qwen3.5:4b"}],
        running=[{"name": "qwen3.5:4b"}],
    )

    status = check_ollama("http://192.168.1.50:11434/", "qwen3.5:4b", timeout=2.5)

    assert status == {
        "reachable": True,
        "version": "0.34.0",
        "has_model": True,
        "loaded": ["qwen3.5:4b"],
        "error": "",
    }
    assert [url for url, _ in seen] == [
        "http://192.168.1.50:11434/api/version",
        "http://192.168.1.50:11434/api/tags",
        "http://192.168.1.50:11434/api/ps",
    ]
    assert all(timeout == 2.5 for _, timeout in seen)


def test_check_ollama_explains_missing_model_on_remote_server(monkeypatch):
    fake_server(monkeypatch, tags=[{"name": "gemma4:e4b-mlx"}], running=[])

    status = check_ollama("http://192.168.1.50:11434", "qwen3.5:4b")

    assert status["reachable"] is True
    assert status["has_model"] is False
    assert status["error"] == (
        "Model qwen3.5:4b is not on the Ollama server at http://192.168.1.50:11434. "
        "Run `ollama pull qwen3.5:4b` on that server."
    )


@pytest.mark.parametrize(
    ("url", "hint"),
    [
        ("http://127.0.0.1:11434", "brew services start ollama"),
        ("http://192.168.1.50:11434", "OLLAMA_HOST=0.0.0.0:11434"),
    ],
)
def test_check_ollama_explains_unreachable_server(monkeypatch, url, hint):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    status = check_ollama(url, "qwen3.5:4b")

    assert status["reachable"] is False
    assert url in status["error"]
    assert hint in status["error"]
