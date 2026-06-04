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

# 로그용: 이전 세그먼트 끝 ~ 다음 시작 갭이 이 값 이상이면 [STT공백] 로그
GAP_LOG_THRESHOLD_SEC = float(os.getenv("STT_GAP_LOG_SEC", "90"))

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
]


def clean_segment_text(text: str) -> str:
    """환각 부분 문자열 제거 및 공백 정리."""
    t = text.strip()
    for phrase in HALLUCINATION_PHRASES:
        t = t.replace(phrase, " ")
    for pattern, replacement in PHRASE_REPEAT_RULES:
        t = re.sub(pattern, replacement, t)
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

        if duration >= min_duration and cps < max_cps:
            dropped += 1
            logger.info(
                f"[STT공백환각] 제거 ({duration:.0f}초, {cps:.1f}자/초): '{text[:50]}'"
            )
            continue
        out.append(seg)

    return out, dropped


def log_large_gaps(segments: List[Dict[str, Any]]) -> int:
    """세그먼트 사이 긴 무음 구간 로그 (타임라인 점검용)."""
    count = 0
    for i in range(1, len(segments)):
        prev_end = float(segments[i - 1].get("end", 0))
        cur_start = float(segments[i].get("start", 0))
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
        "gaps_logged": log_large_gaps(segments),
        "sparse_dropped": 0,
        "long_split": 0,
    }
    segments, stats["sparse_dropped"] = filter_sparse_long_segments(segments)
    segments, stats["long_split"] = split_long_segments(segments)
    return segments, stats
