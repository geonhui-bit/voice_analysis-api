"""
STT 이후 공통 분석 파이프라인.
FastAPI(main.py)와 Celery(audio_handler_v2)가 동일 로직을 사용하도록 동기화.
"""
from typing import Any, Dict, List

from app.services.prosody_analyzer import analyze_prosody_full, format_prosody_inline_tag
from app.services.guard_filter import (
    identify_speaker_roles,
    compute_speaker_profiles,
    detect_dangers_from_audio,
    build_enriched_transcript,
)
from app.services.transcript_builder import (
    build_analysis_meta,
    build_prosody_transcript,
    serialize_segment_prosody,
)


def run_post_stt_pipeline(
    audio_path: str,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    운율(세그먼트별) → 화자 역할·프로필 → 위험 감지 → LLM용 트랜스크립트 조립.
    ML 감정 모델은 사용하지 않음 (emotion_data 빈 dict).
    """
    emotion_data: Dict[str, Any] = {}

    segment_prosody, prosody_data = analyze_prosody_full(audio_path, segments)
    speaker_roles = identify_speaker_roles(segments, emotion_data, prosody_data)
    speaker_profiles = compute_speaker_profiles(segments, speaker_roles)
    danger_detection = detect_dangers_from_audio(segments, emotion_data, prosody_data)
    enriched_transcript = build_enriched_transcript(segments, speaker_roles)

    duration_sec = int(segments[-1].get("end", 0)) if segments else 0
    analysis_meta = build_analysis_meta(
        speaker_roles, prosody_data, danger_detection, speaker_profiles
    )
    prosody_transcript = build_prosody_transcript(
        segments, segment_prosody, speaker_roles, format_prosody_inline_tag
    )

    return {
        "segments": segments,
        "emotion_data": emotion_data,
        "segment_prosody": segment_prosody,
        "prosody_data": prosody_data,
        "speaker_roles": speaker_roles,
        "speaker_profiles": speaker_profiles,
        "danger_detection": danger_detection,
        "transcript_enriched": enriched_transcript,
        "prosody_transcript": prosody_transcript,
        "analysis_meta": analysis_meta,
        "duration_sec": duration_sec,
    }
