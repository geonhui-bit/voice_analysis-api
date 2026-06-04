"""LLM 입력용 트랜스크립트·메타데이터 조립"""
from typing import Dict, List, Any, Optional, Callable

from app.services.guard_filter import role_for_speaker


def build_analysis_meta(
    speaker_roles: Dict[str, str],
    prosody_data: Dict[str, Dict[str, Any]],
    danger_detection: Optional[Dict] = None,
) -> str:
    lines = ["[화자별 운율 종합 (baseline 기준)]"]
    for speaker, role in speaker_roles.items():
        pros = prosody_data.get(speaker, {})
        voice_desc = pros.get("voice_description", "분석 불가")
        bp = pros.get("baseline_pitch", 0)
        lines.append(f"- {role}({speaker}): baseline {bp:.0f}Hz, {voice_desc}")
    if danger_detection and danger_detection.get("detected"):
        lines.append(f"- 위험 플래그: {', '.join(danger_detection['flags'])}")
    return "\n".join(lines)


def build_prosody_transcript(
    segments: List[Dict],
    segment_prosody: List[Dict],
    speaker_roles: Dict[str, str],
    format_prosody_tag: Callable[[Dict[str, Any]], str],
) -> str:
    lines = []
    for i, seg in enumerate(segments):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        role = role_for_speaker(speaker, speaker_roles)

        start_str = f"{int(start // 60):02d}:{start % 60:05.1f}"
        end_str = f"{int(end // 60):02d}:{end % 60:05.1f}"

        sp = segment_prosody[i] if i < len(segment_prosody) else None
        if sp and sp.get("prosody"):
            tag = format_prosody_tag(sp["prosody"])
            lines.append(f"[{start_str}~{end_str}] [{role}] {text}  |운율: {tag}|")
        else:
            lines.append(f"[{start_str}~{end_str}] [{role}] {text}")
    return "\n".join(lines)


def serialize_segment_prosody(segment_prosody: List[Dict]) -> List[Dict]:
    """API 응답용: 세그먼트별 운율 요약"""
    out = []
    for item in segment_prosody:
        pros = item.get("prosody")
        entry = {
            "seg_idx": item.get("seg_idx"),
            "speaker": item.get("speaker"),
            "prosody": pros,
        }
        out.append(entry)
    return out
