"""Word lookup: explain one clicked transcript word in the meaning it has in context.

One Ollama chat request per click. The model answers in a fixed JSON schema, so the
overlay can render a card and the history can store the fields separately.

Manual check without the overlay:
    ./.venv/bin/python -m live_translation.lookup figured "I finally figured out why it broke"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from live_translation.ollama_client import (
    DEFAULT_OLLAMA_URL,
    missing_model_message,
    resolve_ollama_url,
    unreachable_message,
)
from live_translation.text_pipeline import language_name, strip_llm_noise

DEFAULT_LOOKUP_MODEL = "qwen3.5:4b"
LOOKUP_FIELDS = ("lemma", "phrase", "gloss", "explanation")
LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in LOOKUP_FIELDS},
    "required": list(LOOKUP_FIELDS),
}
# Enough surrounding speech for the model to pick the right sense (a bare fragment made
# qwen3.5:4b invent context), but not the whole block of a long monologue.
CONTEXT_WORDS_BEFORE = 30
CONTEXT_WORDS_AFTER = 20
ELLIPSIS = "…"

_RUSSIAN_EXAMPLE = (
    "Example. Word: pulled. Context: he totally [[pulled]] it off in the last round\n"
    '{"lemma": "pull", "phrase": "pull it off", "gloss": "справился, провернул", '
    '"explanation": "Pull it off — успешно сделать что-то трудное. Здесь: он сумел '
    'добиться своего в последнем раунде."}'
)


class WordLookupError(RuntimeError):
    """The lookup could not be completed: Ollama is unreachable or the answer is unusable."""


def context_window(
    text, start, end, words_before=CONTEXT_WORDS_BEFORE, words_after=CONTEXT_WORDS_AFTER
):
    """Trim text to whole words around text[start:end].

    Returns (context, start, end) where the offsets locate the same word inside context.
    Trimmed sides are marked with an ellipsis.
    """
    text = str(text or "")
    if not 0 <= start < end <= len(text):
        raise ValueError(f"word range {start}:{end} is outside the text")
    tokens = [match.span() for match in re.finditer(r"\S+", text)]
    if not tokens:
        raise ValueError("text has no words")
    first_word = next(
        (i for i, (_s, token_end) in enumerate(tokens) if token_end > start), len(tokens) - 1
    )
    last_word = max(
        (i for i, (token_start, _e) in enumerate(tokens) if token_start < end), default=first_word
    )
    lo_idx = max(0, first_word - words_before)
    hi_idx = min(len(tokens) - 1, last_word + words_after)
    lo = min(tokens[lo_idx][0], start)
    hi = max(tokens[hi_idx][1], end)
    prefix = f"{ELLIPSIS} " if lo_idx > 0 else ""
    suffix = f" {ELLIPSIS}" if hi_idx < len(tokens) - 1 else ""
    shift = len(prefix) - lo
    return f"{prefix}{text[lo:hi]}{suffix}", start + shift, end + shift


def mark_word(context, start, end):
    return f"{context[:start]}[[{context[start:end]}]]{context[end:]}"


def lookup_messages(word, marked_context, language_code="ru"):
    language = language_name(language_code)
    system = (
        f"You help a {language} speaker who watches English live streams and videos. They "
        "clicked one word in the speech transcript because they did not catch its meaning. "
        "The clicked word is marked with [[double brackets]] in the context. The transcript "
        "comes from speech recognition: punctuation may be missing and sentences may run "
        "together.\n"
        f"Explain the word in {language} in exactly the meaning it has in this context. Do "
        "not list other meanings. Do not invent facts that are not in the context.\n"
        "Return JSON with these fields:\n"
        '- lemma: the English dictionary form, lowercase, verbs without "to".\n'
        "- phrase: if the clicked word belongs to a phrasal verb, idiom or fixed expression "
        'in the context (for example "right now", "pull it off", "figure out", "miss '
        'something"), give that expression in its dictionary form; otherwise an empty string.\n'
        f"- gloss: the {language} equivalent in this context, 1-4 words.\n"
        f"- explanation: at most two short {language} sentences, under 30 words total, on "
        "what it means here."
    )
    if (language_code or "").lower() == "ru":
        system = f"{system}\n{_RUSSIAN_EXAMPLE}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Word: {word}\nContext: {marked_context}"},
    ]


def _clean_field(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_lookup_response(content, word):
    """Extract the lookup fields from the model answer; raise WordLookupError if unusable."""
    text = strip_llm_noise(str(content or ""))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise WordLookupError(f"model answer is not JSON: {text[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise WordLookupError(f"model answer is not valid JSON: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise WordLookupError("model answer is not a JSON object")
    result = {field: _clean_field(data.get(field)) for field in LOOKUP_FIELDS}
    result["lemma"] = re.sub(r"^to\s+", "", result["lemma"].lower()) or str(word).lower()
    if not result["gloss"] and not result["explanation"]:
        raise WordLookupError("model answer has no explanation")
    return result


class OllamaWordLookup:
    def __init__(
        self,
        model=DEFAULT_LOOKUP_MODEL,
        url=DEFAULT_OLLAMA_URL,
        language="ru",
        temperature=0.2,
        max_tokens=260,
        timeout=120.0,
        keep_alive="30m",
    ):
        self.model = model
        self.language = language
        self.url = url.rstrip("/")
        self.chat_url = self.url + "/api/chat"
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def set_model(self, model):
        if model:
            self.model = model

    def set_language(self, language):
        if language:
            self.language = language

    def lookup(self, text, start, end):
        """Explain text[start:end] as it is used in text. One Ollama request."""
        # Snapshot settings: the UI thread may change them while this runs in a worker.
        model, language = self.model, self.language
        context, ctx_start, ctx_end = context_window(text, start, end)
        word = context[ctx_start:ctx_end]
        payload = {
            "model": model,
            "messages": lookup_messages(word, mark_word(context, ctx_start, ctx_end), language),
            "stream": False,
            "think": False,
            "format": LOOKUP_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": int(self.max_tokens)},
        }
        req = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise WordLookupError(missing_model_message(self.url, model)) from exc
            raise WordLookupError(
                f"Ollama at {self.url} returned HTTP {exc.code} for {model}."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise WordLookupError(unreachable_message(self.url)) from exc
        except ValueError as exc:
            raise WordLookupError("Ollama returned a malformed response") from exc
        fields = parse_lookup_response((body.get("message") or {}).get("content", ""), word)
        return {
            "word": word,
            **fields,
            "context": context,
            "context_start": ctx_start,
            "context_end": ctx_end,
            "language": language,
            "model": model,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explain one English word in context via Ollama.")
    parser.add_argument("word")
    parser.add_argument("context", help="the phrase the word was heard in")
    parser.add_argument("--model", default=DEFAULT_LOOKUP_MODEL)
    parser.add_argument(
        "--url", default=None, help="Ollama server URL (default: $OLLAMA_HOST, else local)"
    )
    parser.add_argument("--lang", default="ru", help="language of the explanation (default: ru)")
    args = parser.parse_args(argv)
    match = re.search(
        rf"(?<![\w'’-]){re.escape(args.word)}(?![\w'’-])", args.context, flags=re.IGNORECASE
    )
    if match is None:
        parser.error(f"{args.word!r} does not occur in the context")
    lookup = OllamaWordLookup(
        model=args.model, url=resolve_ollama_url(args.url, os.environ), language=args.lang
    )
    started = time.monotonic()
    try:
        result = lookup.lookup(args.context, match.start(), match.end())
    except WordLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[lookup] {time.monotonic() - started:.1f}s via {args.model}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
