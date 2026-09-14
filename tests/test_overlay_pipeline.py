import queue

from live_translate_overlay import (
    TRANSLATION_HISTORY_MAX_PAIRS,
    overlap_tail_start,
    post_source_and_enqueue_translation,
    release_mlx_whisper_model,
    translation_history,
)


class RecordingOverlay:
    def __init__(self):
        self.sources = []

    def post_source(self, source, pause_ms=0):
        self.sources.append((source, pause_ms))


def test_finalized_transcript_is_posted_before_translation_queue_drop():
    translation_q = queue.Queue(maxsize=1)
    translation_q.put_nowait({"source": "old untranslated block", "pause_ms": 0.0})
    overlay = RecordingOverlay()

    post_source_and_enqueue_translation(
        translation_q,
        overlay,
        "new finalized transcript",
        pause_ms=1500.0,
        source_language="en",
    )

    assert overlay.sources == [("new finalized transcript", 1500.0)]
    queued = translation_q.get_nowait()
    assert queued == {
        "source": "new finalized transcript",
        "pause_ms": 1500.0,
        "source_language": "en",
    }


def test_overlap_tail_start_drops_overlap_on_clean_vad_cut():
    # Silence-aligned cut: next chunk starts fresh, no re-fed overlap audio.
    assert overlap_tail_start(cut=96000, overlap_frames=12000, clean_cut=True) == 96000


def test_overlap_tail_start_keeps_overlap_on_midspeech_cut():
    # Forced mid-speech cut: keep the overlap tail so a straddling word isn't lost.
    assert overlap_tail_start(cut=96000, overlap_frames=12000, clean_cut=False) == 84000


def test_overlap_tail_start_handles_no_overlap_and_underflow():
    assert overlap_tail_start(cut=5000, overlap_frames=0, clean_cut=False) == 5000
    assert overlap_tail_start(cut=8000, overlap_frames=12000, clean_cut=False) == 0


def test_translation_history_keeps_recent_pairs_newest_last():
    pairs = [("s1", "t1"), ("s2", "t2"), ("s3", "t3")]
    history = translation_history(pairs, exclude_last=False)
    assert history == pairs[-TRANSLATION_HISTORY_MAX_PAIRS:]
    assert history[-1] == ("s3", "t3")


def test_translation_history_excludes_last_pair_when_merging():
    pairs = [("s1", "t1"), ("s2", "t2"), ("s3", "t3")]
    # exclude_last drops the newest pair (it's being merged into the current source)
    history = translation_history(pairs, exclude_last=True)
    assert ("s3", "t3") not in history
    assert history == pairs[:-1][-TRANSLATION_HISTORY_MAX_PAIRS:]


def test_translation_history_bounded_by_chars():
    big = "x" * 5000
    pairs = [("a", "b"), (big, big)]
    history = translation_history(pairs, exclude_last=False)
    # The oversized newest pair alone is allowed, but it must not drag in older ones.
    assert history == [(big, big)]


def test_release_mlx_whisper_model_clears_matching_holder():
    from mlx_whisper.transcribe import ModelHolder

    previous_model = ModelHolder.model
    previous_path = ModelHolder.model_path
    try:
        ModelHolder.model = object()
        ModelHolder.model_path = "mlx-community/whisper-small-mlx"

        assert release_mlx_whisper_model("mlx-community/whisper-small-mlx") is True
        assert ModelHolder.model is None
        assert ModelHolder.model_path is None
    finally:
        ModelHolder.model = previous_model
        ModelHolder.model_path = previous_path


class RecordingLookupOverlay:
    def __init__(self):
        self.results = []

    def post_lookup_result(self, card_id, result=None, error=""):
        self.results.append((card_id, result, error))


class FakeWordLookup:
    def __init__(self, fail=False):
        self.fail = fail
        self.settings_seen = []

    def set_model(self, model):
        self.settings_seen.append(("model", model))

    def set_language(self, language):
        self.settings_seen.append(("language", language))

    def lookup(self, text, start, end):
        if self.fail:
            raise RuntimeError("Ollama is not responding.")
        return {"word": text[start:end], "lemma": "figure", "gloss": "понял"}


def run_lookup_worker_until_result(word_lookup, item):
    import threading
    import time

    from live_translate_overlay import lookup_worker
    from live_translation.translators import LanguageSettings

    lookup_q = queue.Queue()
    lookup_q.put(item)
    overlay = RecordingLookupOverlay()
    stop_event = threading.Event()
    settings = LanguageSettings("en", "ru", ollama_model="qwen3.5:4b")
    worker = threading.Thread(
        target=lookup_worker,
        args=(lookup_q, overlay, word_lookup, settings, stop_event),
        daemon=True,
    )
    worker.start()
    deadline = time.monotonic() + 2.0
    while not overlay.results and time.monotonic() < deadline:
        time.sleep(0.01)
    stop_event.set()
    worker.join(timeout=2.0)
    return overlay.results


def test_lookup_worker_posts_result_with_current_settings():
    word_lookup = FakeWordLookup()
    text = "I finally figured out why"

    results = run_lookup_worker_until_result(
        word_lookup, {"card_id": 7, "text": text, "start": 10, "end": 17}
    )

    assert results == [(7, {"word": "figured", "lemma": "figure", "gloss": "понял"}, "")]
    assert word_lookup.settings_seen == [("model", "qwen3.5:4b"), ("language", "ru")]


def test_lookup_worker_posts_error_and_keeps_running():
    results = run_lookup_worker_until_result(
        FakeWordLookup(fail=True), {"card_id": 3, "text": "hello", "start": 0, "end": 5}
    )

    assert results == [(3, None, "Ollama is not responding.")]
