"""
업그레이드된 오디오 처리 서비스 (v2)
WhisperX + 감정분석 + 프로소디 + 가드레일 통합
"""
import time
from pathlib import Path
from typing import Dict, Any
from app.config import settings
from app.utils.logger import get_logger
from app.utils.file_utils import convert_to_wav, get_audio_duration_seconds
from app.services.whisperx_loader import transcribe_with_diarization
from app.services.emotion_analyzer import analyze_emotions_by_speaker
from app.services.prosody_analyzer import analyze_prosody_by_speaker
from app.services.guard_filter import (
    identify_speaker_roles,
    detect_dangers_from_audio,
    build_enriched_transcript
)

logger = get_logger(__name__)


def process_audio_file_v2(input_path: Path, processing_id: str) -> Dict[str, Any]:
    """
    전체 오디오 분석 파이프라인 (v2)

    단계:
    1. WAV 변환
    2. WhisperX (STT + 화자분리 + 타임스탬프)
    3. 감정분석 (화자별)
    4. 프로소디 분석 (화자별)
    5. 가드레일 (역할 매핑 + 위험 감지)
    6. LLM용 enriched transcript 생성
    """
    logger.info(f"[{processing_id}] v2 오디오 처리 시작: {input_path.name}")
    start_time = time.perf_counter()

    try:
        # 1. 오디오 메타데이터
        duration_seconds = get_audio_duration_seconds(input_path)

        if duration_seconds > settings.MAX_AUDIO_DURATION:
            raise ValueError(
                f"오디오가 너무 깁니다 ({duration_seconds:.0f}초). "
                f"최대 {settings.MAX_AUDIO_DURATION}초"
            )

        # 2. WAV 변환
        wav_path = settings.SHARED_CONVERTED_DIR / f"converted_{processing_id}.wav"
        convert_to_wav(input_path, wav_path)
        wav_path_str = str(wav_path)

        # 3. GCS 업로드
        from app.utils.gcs import upload_to_gcs
        converted_gcs_path = f"audio/converted_{processing_id}.wav"
        audio_gcs_url = upload_to_gcs(wav_path, converted_gcs_path)

        # 4. WhisperX STT + 화자분리
        stt_start = time.perf_counter()
        whisperx_result = transcribe_with_diarization(wav_path_str)
        stt_time = time.perf_counter() - stt_start
        logger.info(f"[{processing_id}] STT+화자분리 완료 ({stt_time:.2f}초)")

        segments = whisperx_result["segments"]

        # 5. 감정분석 (화자별)
        emo_start = time.perf_counter()
        emotion_data = analyze_emotions_by_speaker(wav_path_str, segments)
        emo_time = time.perf_counter() - emo_start
        logger.info(f"[{processing_id}] 감정분석 완료 ({emo_time:.2f}초)")

        # 6. 프로소디 분석 (화자별)
        pros_start = time.perf_counter()
        prosody_data = analyze_prosody_by_speaker(wav_path_str, segments)
        pros_time = time.perf_counter() - pros_start
        logger.info(f"[{processing_id}] 프로소디 분석 완료 ({pros_time:.2f}초)")

        # 7. 가드레일: 화자 역할 매핑
        speaker_roles = identify_speaker_roles(segments, emotion_data, prosody_data)

        # 8. 가드레일: 위험요소 사전 감지
        danger_detection = detect_dangers_from_audio(segments, emotion_data, prosody_data)

        # 9. LLM용 enriched transcript 생성
        enriched_transcript = build_enriched_transcript(segments, speaker_roles)

        # 10. 기존 호환 plain transcript (화자 라벨 없는 버전)
        plain_transcript = whisperx_result["transcript_plain"]

        total_time = time.perf_counter() - start_time
        logger.info(f"[{processing_id}] v2 전체 처리 완료 ({total_time:.2f}초)")

        return {
            "success": True,
            "processing_id": processing_id,
            "duration_seconds": duration_seconds,
            # 트랜스크립트 (3종)
            "transcript": plain_transcript,
            "transcript_labeled": whisperx_result["transcript_labeled"],
            "transcript_enriched": enriched_transcript,
            # 분석 결과
            "segments": segments,
            "speaker_roles": speaker_roles,
            "emotion_data": emotion_data,
            "prosody_data": prosody_data,
            "danger_detection": danger_detection,
            # 메타
            "wav_path": wav_path_str,
            "audio_gcs_url": audio_gcs_url,
            "timing": {
                "stt": round(stt_time, 2),
                "emotion": round(emo_time, 2),
                "prosody": round(pros_time, 2),
                "total": round(total_time, 2)
            }
        }

    except Exception as e:
        logger.error(f"[{processing_id}] v2 오디오 처리 실패: {e}")
        return {
            "success": False,
            "processing_id": processing_id,
            "error": str(e)
        }
