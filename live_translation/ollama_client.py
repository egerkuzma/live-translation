"""Where the Ollama server lives and whether it is ready.

The overlay can use Ollama on this Mac or on another machine in the LAN. The URL comes
from --ollama-url, then the OLLAMA_HOST environment variable (the one the `ollama` CLI
reads), then the local default.
"""

import json
import urllib.parse
import urllib.request

DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_URL = f"http://127.0.0.1:{DEFAULT_OLLAMA_PORT}"
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def resolve_ollama_url(flag_value=None, environ=None):
    """Build the server URL the same way the ollama CLI reads OLLAMA_HOST.

    No scheme means http on port 11434; an explicit http:// or https:// without a port
    means 80 or 443. 0.0.0.0 is a bind address, so it is replaced with 127.0.0.1.
    """
    environ = {} if environ is None else environ
    raw = str(flag_value or environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        return DEFAULT_OLLAMA_URL
    default_port = DEFAULT_OLLAMA_PORT
    if "://" in raw:
        scheme = raw.split("://", 1)[0].lower()
        default_port = {"http": 80, "https": 443}.get(scheme, DEFAULT_OLLAMA_PORT)
    else:
        raw = f"http://{raw}"
    parts = urllib.parse.urlsplit(raw)
    host = parts.hostname or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    port = parts.port or default_port
    return f"{parts.scheme}://{host}:{port}{parts.path.rstrip('/')}"


def is_local_url(url):
    return (urllib.parse.urlsplit(url).hostname or "") in _LOCAL_HOSTS


def unreachable_message(url):
    if is_local_url(url):
        return f"Ollama at {url} is not responding. Start it: `brew services start ollama`."
    return (
        f"Ollama at {url} is not responding. Check that the server is on and listens "
        "on the network (OLLAMA_HOST=0.0.0.0:11434 on that machine)."
    )


def missing_model_message(url, model):
    where = "" if is_local_url(url) else " on that server"
    return f"Model {model} is not on the Ollama server at {url}. Run `ollama pull {model}`{where}."


def _get_json(url, timeout):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama(url, model, timeout=3.0):
    """Ask the server for its version, pulled models and loaded models.

    Returns a dict with reachable, version, has_model, loaded (model names) and error,
    where error is a ready-to-print hint or an empty string when everything is fine.
    """
    base = url.rstrip("/")
    status = {"reachable": False, "version": "", "has_model": False, "loaded": [], "error": ""}
    try:
        version = _get_json(f"{base}/api/version", timeout).get("version")
        pulled = _get_json(f"{base}/api/tags", timeout).get("models") or []
        running = _get_json(f"{base}/api/ps", timeout).get("models") or []
    except (OSError, ValueError, AttributeError):
        status["error"] = unreachable_message(url)
        return status
    names = {name for m in pulled for name in (m.get("name"), m.get("model")) if name}
    status.update(
        reachable=True,
        version=str(version or ""),
        has_model=model in names,
        loaded=[
            m.get("name") or m.get("model") for m in running if m.get("name") or m.get("model")
        ],
    )
    if not status["has_model"]:
        status["error"] = missing_model_message(url, model)
    return status
