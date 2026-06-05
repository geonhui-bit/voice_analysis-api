"""
STT 후처리: Whisper 환각 문구, 긴 공백·초장 세그먼트 정리.
"""
import os
import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# 인접 세그먼트 병합 시 허용 최대 갭(초) — 이보다 크면 병합 금지
MAX_MERGE_GAP_SEC = float(os.getenv("STT_MAX_MERGE_GAP_SEC", "30"))

# 초장 세그먼트 분할 기준(초)
MAX_SEGMENT_DURATION_SEC = float(os.getenv("STT_MAX_SEGMENT_SEC", "90"))

# 긴 구간인데 글자 수가 적으면 공백 환각으로 간주
SPARSE_MIN_DURATION_SEC = float(os.getenv("STT_SPARSE_MIN_DURATION_SEC", "45"))
SPARSE_MAX_CHARS_PER_SEC = float(os.getenv("STT_SPARSE_MAX_CHARS_PER_SEC", "1.2"))
# 루프 축약 후에도 유지할 최소 글자 수(공백 제외) — 실제 발화 보호
# 이 글자 수 미만 + 저밀도일 때만 제거 (짧은 실발화는 유지)
SPARSE_MIN_KEEP_CHARS = int(os.getenv("STT_SPARSE_MIN_KEEP_CHARS", "10"))

# 로그·무음 마커: 이전 세그먼트 끝 ~ 다음 시작 갭이 이 값 이상
GAP_LOG_THRESHOLD_SEC = float(os.getenv("STT_GAP_LOG_SEC", "90"))
GAP_MARKER_THRESHOLD_SEC = float(
    os.getenv("STT_GAP_MARKER_SEC", os.getenv("STT_GAP_LOG_SEC", "90"))
)

# 무음 직후·혼합 발화: 역할 기반 재분할 (12-2)
POST_SILENCE_RESPLIT_MIN_SEC = float(os.getenv("STT_POST_SILENCE_RESPLIT_SEC", "45"))
POST_SILENCE_MIXED_MIN_SCORE = int(os.getenv("STT_POST_SILENCE_MIXED_SCORE", "2"))

_MANAGER_RESPLIT_HINTS = (
    "어르신", "어머니", "드셔", "드시", "하세요", "하셔", "잡숴", "먹어도",
    "기력회복", "밥을", "식사 잘", "나가세요", "드릴게", "괜찮아 그거",
    "충분해", "끓는", "드시면",
)
_SENIOR_RESPLIT_HINTS = (
    "내가", "나는", "나도", "우리 사돈", "돌아가", "사돈", "며느리",
    "자식", "아버지", "옛날", "이사 오", "베트남", "4돈", "맹니",
    "배불런", "여태까지", "남아놨", "죽었어",
)

# 긴 무음 구간 표시용 가상 화자 (pyannote ID 아님)
SPEAKER_SILENCE_GAP = "SILENCE_GAP"

# 긴 문구부터 제거 (부분 중복 방지)
HALLUCINATION_PHRASES = [
    "자막 제공 및 해 주신 모든 분들께",
    "자막 제공 및 해 주신 모든",
    "및 해 주신 모든 분들께",
    "및 해 주신 모든",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "자막 제공",
    "및 해",
    "이 시각 세계였습니다",
    "이 시각 세계",
]

# 세그먼트 전체가 이것뿐이면 삭제
JUNK_ONLY_TEXTS = {
    "",
    ".",
    "..",
    "...",
    "및 해",
    "자막 제공",
    "안녕하세요",
    "안녕하세요.",
}

# 한 세그먼트 안에서 연속 반복 축약 (실험에서 확인된 패턴)
PHRASE_REPEAT_RULES = [
    (r"(?:갔다\s*왔다[\s,]*){3,}", "갔다 왔다 "),
    (r"(?:왔다\s*갔다[\s,]*){3,}", "갔다 왔다 "),
    (r"(?:잘\s*)?(?:가르쳐야\s*하는데[\s,.]*){3,}", "가르쳐야 하는데 "),
    (r"(?:가르쳐야\s*하는데[\s,.]*){3,}", "가르쳐야 하는데 "),
    (r"(?:이\s*시각\s*세계였습니다[.\s]*){2,}", ""),
    (r"(?:이\s*시각\s*세계[.\s]*){2,}", ""),
]

# 연속 단어/구 반복 루프 (분할 후 세그먼트 내부)
MIN_LOOP_UNIT_WORDS = 2
MAX_LOOP_UNIT_WORDS = 14
MIN_LOOP_REPEATS = 3

# 9단계: 방문요양 맥락 STT 동음이의·유사음 오타 (보수적 적용)
TYPO_CORRECT_ENABLED = os.getenv("STT_TYPO_CORRECT", "1").lower() in (
    "1",
    "true",
    "yes",
)

_SENIOR_CARE_CONTEXT = re.compile(
    r"(?:어르신|못하시|영양|식사|잡숴|기력|드시|먹으|밥|국물|약|병원|통풍|간병|어르신들)",
    re.IGNORECASE,
)
_CHILD_CONTEXT = re.compile(
    r"(?:어린이집|유치원|어린이들|어린이가|어린이와|아이들|유아)",
)
_EXAM_STUDY_CONTEXT = re.compile(
    r"(?:시험공부|시험지|수능|입시|학교\s*시험|시험\s*보)",
)

# 문맥 없이도 안전한 고정 치환 (긴 구문 우선)
TYPO_PHRASE_FIXES = [
    ("어린이 식사", "어르신 식사"),
    ("어린이 식사를", "어르신 식사를"),
    ("도룩절로", "도수치료로"),
    ("도룩절", "도수치료"),
    ("엑스레일", "엑스레이"),
    ("경원경원", "경원"),
    ("관호사", "간호사"),
    ("우산 도우로", "우산동으로"),
    ("일어먹는", "일어 먹는"),
    ("거는는", "거는"),
    ("인명하게", "인명사하게"),
]

# 시험→식사: 식사 행위와 함께 나올 때만
TYPO_REGEX_FIXES = [
    (r"시험을\s*먹", "식사를 먹"),
    (r"시험을\s*드", "식사를 드"),
    (r"시험\s*먹으", "식사 먹으"),
    (r"시험\s*잡숴", "식사 잡숴"),
    (r"시험을\s*먹으라", "식사를 먹으라"),
    (r"이제\s*시험을\s*먹", "이제 식사를 먹"),
    (r"그래\s*이제\s*시험을", "그래 이제 식사를"),
]


def apply_typo_corrections(text: str) -> str:
    """
    방문요양 대화에서 확인된 STT 오타 보정.
    어린이↔어르신, 시험↔식사 등 — 아동·학업 맥락이면 건너뜀.
    """
    if not TYPO_CORRECT_ENABLED or not text.strip():
        return text

    t = text
    for old, new in TYPO_PHRASE_FIXES:
        if old in t:
            t = t.replace(old, new)

    if "어린이" in t and not _CHILD_CONTEXT.search(t):
        if _SENIOR_CARE_CONTEXT.search(t) or "못하시" in t or "영양" in t:
            t = re.sub(r"어린이", "어르신", t)

    if "시험" in t and not _EXAM_STUDY_CONTEXT.search(t):
        if _SENIOR_CARE_CONTEXT.search(t) or re.search(
            r"(?:먹으|드시|밥|식사|잡숴|국물|끓이|고기)", t
        ):
            for pattern, repl in TYPO_REGEX_FIXES:
                t = re.sub(pattern, repl, t)

    return t


def collapse_word_sequence_loops(text: str) -> str:
    """
    동일한 단어 시퀀스가 3회 이상 연속 반복되면 1회만 남김.
    예: '갔다 왔다' x20, '가르쳐야 하는데' x10
    """
    words = text.split()
    if len(words) < MIN_LOOP_UNIT_WORDS * MIN_LOOP_REPEATS:
        return text

    max_unit = min(MAX_LOOP_UNIT_WORDS, len(words) // MIN_LOOP_REPEATS)
    for unit_len in range(max_unit, MIN_LOOP_UNIT_WORDS - 1, -1):
        for i in range(len(words) - unit_len * MIN_LOOP_REPEATS + 1):
            unit = words[i : i + unit_len]
            count = 1
            j = i + unit_len
            while j + unit_len <= len(words) and words[j : j + unit_len] == unit:
                count += 1
                j += unit_len
            if count >= MIN_LOOP_REPEATS:
                collapsed = words[:i] + unit + words[j:]
                return collapse_word_sequence_loops(" ".join(collapsed))
    return text


def clean_segment_text(text: str) -> str:
    """환각 부분 문자열 제거 및 공백 정리."""
    t = text.strip()
    for phrase in HALLUCINATION_PHRASES:
        t = t.replace(phrase, " ")
    for pattern, replacement in PHRASE_REPEAT_RULES:
        t = re.sub(pattern, replacement, t)
    t = collapse_word_sequence_loops(t)
    t2 = apply_typo_corrections(t)
    if t2 != t:
        logger.debug(f"[STT오타] '{t[:40]}' → '{t2[:40]}'")
    t = t2
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[\s.,·…]+$", "", t)
    return t


def filter_hallucination_segments(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    세그먼트 텍스트 정리. 정리 후 내용이 없으면 세그먼트 제거.

    Returns:
        (정리된 segments, {"text_cleaned": n, "segments_dropped": n})
    """
    stats = {"text_cleaned": 0, "segments_dropped": 0}
    out: List[Dict[str, Any]] = []

    for seg in segments:
        raw = seg.get("text", "").strip()
        cleaned = clean_segment_text(raw)

        if cleaned != raw:
            stats["text_cleaned"] += 1

        if cleaned in JUNK_ONLY_TEXTS:
            stats["segments_dropped"] += 1
            logger.debug(f"[STT환각] 세그먼트 제거: '{raw[:40]}'")
            continue

        if not cleaned:
            stats["segments_dropped"] += 1
            continue

        new_seg = dict(seg)
        new_seg["text"] = cleaned
        out.append(new_seg)

    return out, stats


def _chars_per_sec(text: str, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return len(text.replace(" ", "")) / duration_sec


def _split_text_evenly(text: str, parts: int) -> List[str]:
    words = text.split()
    if parts <= 1 or len(words) < parts:
        return [text]
    chunk = max(1, len(words) // parts)
    result = []
    for i in range(0, len(words), chunk):
        piece = " ".join(words[i : i + chunk]).strip()
        if piece:
            result.append(piece)
    return result or [text]


def split_long_segments(
    segments: List[Dict[str, Any]],
    max_duration: float = MAX_SEGMENT_DURATION_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """초장 세그먼트를 문장 단위(불가 시 균등 분할)로 나눔."""
    out: List[Dict[str, Any]] = []
    split_count = 0

    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = seg.get("text", "").strip()
        duration = end - start

        if duration <= max_duration or not text:
            out.append(dict(seg))
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.?!？！])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            n_parts = max(2, int(duration / max_duration) + 1)
            sentences = _split_text_evenly(text, n_parts)

        total_chars = sum(len(s) for s in sentences) or 1
        t = start
        for sent in sentences:
            frac = len(sent) / total_chars
            seg_end = min(t + duration * frac, end)
            if seg_end <= t:
                continue
            piece = dict(seg)
            piece.update({"start": t, "end": seg_end, "text": sent})
            out.append(piece)
            t = seg_end
        split_count += 1
        logger.info(
            f"[STT분할] {duration:.0f}초 세그먼트 → {len(sentences)}개: '{text[:40]}...'"
        )

    return out, split_count


def filter_sparse_long_segments(
    segments: List[Dict[str, Any]],
    min_duration: float = SPARSE_MIN_DURATION_SEC,
    max_cps: float = SPARSE_MAX_CHARS_PER_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """긴 구간 + 텍스트 거의 없음 → 공백 환각 세그먼트 제거."""
    out: List[Dict[str, Any]] = []
    dropped = 0

    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = seg.get("text", "").strip()
        duration = end - start
        cps = _chars_per_sec(text, duration)

        char_count = len(text.replace(" ", ""))
        if (
            duration >= min_duration
            and cps < max_cps
            and char_count < SPARSE_MIN_KEEP_CHARS
        ):
            dropped += 1
            logger.info(
                f"[STT공백환각] 제거 ({duration:.0f}초, {cps:.1f}자/초): '{text[:50]}'"
            )
            continue
        out.append(seg)

    return out, dropped


def _is_speech_segment(seg: Dict[str, Any]) -> bool:
    return not seg.get("is_silence_gap") and seg.get("speaker") != SPEAKER_SILENCE_GAP


def _format_gap_duration(gap_sec: float) -> str:
    if gap_sec >= 3600:
        return f"약 {gap_sec / 3600:.1f}시간"
    if gap_sec >= 60:
        return f"약 {int(round(gap_sec / 60))}분"
    return f"{int(round(gap_sec))}초"


def _cluster_speech_segments(
    speech: List[Dict[str, Any]],
    min_gap: float,
) -> List[List[Dict[str, Any]]]:
    """인접 갭이 min_gap 미만이면 같은 발화 블록으로 묶음."""
    if not speech:
        return []
    clusters: List[List[Dict[str, Any]]] = [[speech[0]]]
    for seg in speech[1:]:
        gap = float(seg.get("start", 0)) - float(clusters[-1][-1].get("end", 0))
        if gap < min_gap:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])
    return clusters


def remove_segments_in_long_gaps(
    segments: List[Dict[str, Any]],
    min_gap: float = GAP_MARKER_THRESHOLD_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    두 발화 블록 사이 긴 무음 안에 끼인 STT 세그먼트(환각) 제거.
    """
    speech = sorted(
        [s for s in segments if _is_speech_segment(s)],
        key=lambda s: float(s.get("start", 0)),
    )
    clusters = _cluster_speech_segments(speech, min_gap)
    if len(clusters) < 2:
        return segments, 0

    drop_keys: set = set()
    for i in range(len(clusters)):
        for j in range(i + 2, len(clusters)):
            prev_end = float(clusters[i][-1].get("end", 0))
            next_start = float(clusters[j][0].get("start", 0))
            void = next_start - prev_end
            if void < min_gap:
                continue
            for k in range(i + 1, j):
                for mid in clusters[k]:
                    key = (float(mid.get("start", 0)), float(mid.get("end", 0)))
                    drop_keys.add(key)
                    logger.info(
                        f"[STT무음환각] 제거 ({void:.0f}초 공백 내): "
                        f"'{mid.get('text', '')[:50]}'"
                    )

    if not drop_keys:
        return segments, 0

    out = [
        s
        for s in segments
        if not _is_speech_segment(s)
        or (float(s.get("start", 0)), float(s.get("end", 0))) not in drop_keys
    ]
    return out, len(drop_keys)


def insert_silence_gap_markers(
    segments: List[Dict[str, Any]],
    min_gap: float = GAP_MARKER_THRESHOLD_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """긴 무음 구간에 [무음] 마커 세그먼트 삽입 (타임라인·LLM용)."""
    speech = sorted(
        [dict(s) for s in segments if _is_speech_segment(s)],
        key=lambda s: float(s.get("start", 0)),
    )
    if not speech:
        return [], 0

    out: List[Dict[str, Any]] = []
    marker_count = 0

    for i, seg in enumerate(speech):
        if i > 0:
            prev_end = float(speech[i - 1].get("end", 0))
            cur_start = float(seg.get("start", 0))
            gap = cur_start - prev_end
            if gap >= min_gap:
                label = f"(무음 {_format_gap_duration(gap)})"
                out.append(
                    {
                        "start": prev_end,
                        "end": cur_start,
                        "speaker": SPEAKER_SILENCE_GAP,
                        "text": label,
                        "is_silence_gap": True,
                    }
                )
                marker_count += 1
                logger.info(
                    f"[STT무음] 마커 삽입 {gap:.0f}초 ({prev_end:.1f}s → {cur_start:.1f}s)"
                )
        out.append(seg)

    return out, marker_count


def _role_scores_for_resplit(text: str) -> Tuple[int, int]:
    """(매니저 점수, 시니어 점수) — 한 세그먼트 텍스트."""
    m = sum(1 for kw in _MANAGER_RESPLIT_HINTS if kw in text)
    s = sum(1 for kw in _SENIOR_RESPLIT_HINTS if kw in text)
    if ("어르신" in text or "어머니" in text) and not any(
        x in text for x in ("내가", "나는", "나도", "우리 사돈")
    ):
        m += 2
    if any(x in text for x in ("내가", "나는", "나도")):
        s += 2
    return m, s


def _is_mixed_role_text(text: str) -> bool:
    m, s = _role_scores_for_resplit(text)
    return m >= POST_SILENCE_MIXED_MIN_SCORE and s >= POST_SILENCE_MIXED_MIN_SCORE


def _infer_chunk_role(text: str) -> str:
    m, s = _role_scores_for_resplit(text)
    if m > s:
        return "매니저"
    if s > m:
        return "시니어"
    return ""


def _sentence_chunks_for_resplit(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.?!？！…])\s+", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[요다죠네])\s+", text) if p.strip()]
    return parts or [text]


def _group_chunks_by_role(chunks: List[str]) -> List[Dict[str, str]]:
    """연속 문장을 역할별로 묶음."""
    groups: List[Dict[str, str]] = []
    for ch in chunks:
        role = _infer_chunk_role(ch)
        if groups and role and role == groups[-1]["role"]:
            groups[-1]["text"] = groups[-1]["text"] + " " + ch
        elif groups and not role:
            groups[-1]["text"] = groups[-1]["text"] + " " + ch
        else:
            groups.append({"text": ch, "role": role})
    if not groups:
        return [{"text": " ".join(chunks), "role": ""}]

    roles = [g["role"] for g in groups if g["role"]]
    fallback = roles[0] if roles else "시니어"
    for g in groups:
        if not g["role"]:
            g["role"] = fallback
    return groups


def _should_resplit_segment(
    seg: Dict[str, Any],
    prev_seg: Any,
) -> bool:
    if not _is_speech_segment(seg):
        return False
    start = float(seg.get("start", 0))
    end = float(seg.get("end", 0))
    duration = end - start
    text = seg.get("text", "").strip()
    if not text:
        return False

    prev_silence = prev_seg is not None and (
        prev_seg.get("is_silence_gap")
        or prev_seg.get("speaker") == SPEAKER_SILENCE_GAP
    )
    long_after_silence = prev_silence and duration >= POST_SILENCE_RESPLIT_MIN_SEC
    return long_after_silence and _is_mixed_role_text(text)


def resplit_mixed_post_silence_segments(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    무음 마커 직후 등에서 pyannote가 한 화자로 붙인 혼합 발화를
    문장 단위 역할 추론으로 나누고 inferred_role 부여 (운율·LLM 표기용).
    """
    out: List[Dict[str, Any]] = []
    resplit_count = 0

    for i, seg in enumerate(segments):
        if not _is_speech_segment(seg):
            out.append(dict(seg))
            continue

        prev = segments[i - 1] if i > 0 else None
        if not _should_resplit_segment(seg, prev):
            out.append(dict(seg))
            continue

        text = seg.get("text", "").strip()
        chunks = _sentence_chunks_for_resplit(text)
        groups = _group_chunks_by_role(chunks)
        if len(groups) <= 1:
            out.append(dict(seg))
            continue

        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        duration = end - start
        total_chars = sum(len(g["text"]) for g in groups) or 1
        t = start
        for g in groups:
            frac = len(g["text"]) / total_chars
            seg_end = min(t + duration * frac, end)
            if seg_end <= t:
                continue
            piece = dict(seg)
            piece.update({
                "start": t,
                "end": seg_end,
                "text": g["text"],
                "inferred_role": g["role"],
            })
            out.append(piece)
            t = seg_end
        resplit_count += 1
        logger.info(
            f"[STT화자분리] 혼합 세그먼트 → {len(groups)}개 "
            f"({duration:.0f}초, 무음 직후={prev and prev.get('is_silence_gap')}): "
            f"'{text[:40]}...'"
        )

    return out, resplit_count


def log_large_gaps(segments: List[Dict[str, Any]]) -> int:
    """세그먼트 사이 긴 무음 구간 로그 (타임라인 점검용)."""
    count = 0
    speech = [s for s in segments if _is_speech_segment(s)]
    for i in range(1, len(speech)):
        prev_end = float(speech[i - 1].get("end", 0))
        cur_start = float(speech[i].get("start", 0))
        gap = cur_start - prev_end
        if gap >= GAP_LOG_THRESHOLD_SEC:
            count += 1
            logger.info(
                f"[STT공백] {gap:.0f}초 무음 구간 "
                f"({prev_end:.1f}s → {cur_start:.1f}s)"
            )
    return count


def repair_segments_gaps_and_length(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    5단계: 공백 로그 → 저밀도 장세그먼트 제거 → 초장 세그먼트 분할.
    """
    stats = {
        "gaps_logged": 0,
        "sparse_dropped": 0,
        "long_split": 0,
        "loop_cleaned": 0,
        "loop_dropped": 0,
        "gap_hallucination_dropped": 0,
        "silence_markers": 0,
        "post_silence_resplit": 0,
    }
    # 분할 → 루프 재정리 → 저밀도 제거 → 무음 내 환각 제거 → 무음 마커
    segments, stats["long_split"] = split_long_segments(segments)
    segments, loop_stats = reapply_text_cleanup_after_repair(segments)
    stats["loop_cleaned"] = loop_stats["text_cleaned"]
    stats["loop_dropped"] = loop_stats["segments_dropped"]
    segments, stats["sparse_dropped"] = filter_sparse_long_segments(segments)
    segments, stats["gap_hallucination_dropped"] = remove_segments_in_long_gaps(
        segments
    )
    stats["gaps_logged"] = log_large_gaps(segments)
    segments, stats["silence_markers"] = insert_silence_gap_markers(segments)
    segments, stats["post_silence_resplit"] = resplit_mixed_post_silence_segments(
        segments
    )
    return segments, stats


def reapply_text_cleanup_after_repair(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    6단계: 장세그먼트 분할(5c) 후 세그먼트 내부 반복·환각 문구 재정리.
    분할 전(5b)에만 정리하면 분할 결과에 루프가 다시 남을 수 있음.
    """
    stats = {"text_cleaned": 0, "segments_dropped": 0}
    out: List[Dict[str, Any]] = []

    for seg in segments:
        raw = seg.get("text", "").strip()
        cleaned = clean_segment_text(raw)
        if cleaned != raw:
            stats["text_cleaned"] += 1
        if cleaned in JUNK_ONLY_TEXTS or not cleaned:
            stats["segments_dropped"] += 1
            logger.debug(f"[STT루프] 분할 후 빈 세그먼트 제거: '{raw[:40]}'")
            continue
        new_seg = dict(seg)
        new_seg["text"] = cleaned
        out.append(new_seg)

    if stats["text_cleaned"] or stats["segments_dropped"]:
        logger.info(
            f"[STT루프] 분할 후 재정리: 텍스트 {stats['text_cleaned']}건, "
            f"제거 {stats['segments_dropped']}건 → {len(out)}개"
        )
    return out, stats
