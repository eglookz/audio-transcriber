import transcribe as t


def w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_split_on_pause_and_offset():
    segs = t.segments_from_words([w("Привет,", 0, .5), w("мир.", .6, 1), w("Дальше", 2.5, 3)],
                                 offset=100)
    assert [s["text"] for s in segs] == ["Привет, мир.", "Дальше"]
    assert segs[0]["start"] == 100
    assert segs[1]["end"] == 103


def test_long_run_splits_after_sentence_end():
    words = [w(f"слово{i}.", i * 0.5, i * 0.5 + 0.4) for i in range(40)]  # 20 s, no pauses
    segs = t.segments_from_words(words, 0)
    assert len(segs) >= 2
    assert all(s["end"] - s["start"] <= 30 for s in segs)


def test_hard_cap_without_punctuation():
    words = [w("слово", i * 0.5, i * 0.5 + 0.4) for i in range(200)]  # 100 s, no pauses, no dots
    segs = t.segments_from_words(words, 0)
    assert all(s["end"] - s["start"] <= 30 for s in segs)
    assert " ".join(s["text"] for s in segs) == " ".join(["слово"] * 200)


def test_empty_words():
    assert t.segments_from_words([], 0) == []


def test_blank_words_are_skipped():
    segs = t.segments_from_words([w(" ", 0, .1), w("да", .2, .4)], 0)
    assert segs == [{"start": .2, "end": .4, "text": "да"}]


def test_payload_with_words_and_single_segment_uses_words():
    payload = {"text": "Да. Нет.", "segments": [{"start": 0, "end": 5, "text": "Да. Нет."}],
               "words": [w("Да.", 0, .3), w("Нет.", 3, 3.4)]}
    segs = t.payload_segments(payload, offset=10)
    assert [s["text"] for s in segs] == ["Да.", "Нет."]
    assert segs[1]["start"] == 13


def test_payload_with_real_segments_keeps_them():
    payload = {"segments": [{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}],
               "words": [w("a", 0, 1), w("b", 1, 2)]}
    assert [s["text"] for s in t.payload_segments(payload, 0)] == ["a", "b"]


def test_payload_with_empty_words_and_single_segment_keeps_segment():
    payload = {"segments": [{"start": 0, "end": 5, "text": "всё"}], "words": []}
    assert t.payload_segments(payload, 0) == [{"start": 0.0, "end": 5.0, "text": "всё"}]
