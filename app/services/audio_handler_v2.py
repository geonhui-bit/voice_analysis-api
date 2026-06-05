"""
업그레이드된 오디오 처리 서비스 (v2)
WhisperX + 운율(세그먼트) + 가드레일 — main.py /analyze 와 동일 파이프라인
"""
import time
from pathlib import Path
from typing import Dict, Any
from app.config import settings
from app.utils.logger import get_logger
from app.utils.file_utils import convert_to_wav, get_audio_duration_seconds
from app.services.whisperx_loader import transcribe_with_diarization
from app.services.analysis_pipeline import run_post_stt_pipeline
from app.services.transcript_builder import serialize_segment_prosody

logger = get_logger(__name__)


def process_audio_file_v2(input_path: Path, processing_id: str) -> Dict[str, Any]:
    """
    전체 오디오 분석 파이프라인 (v2, FastAPI 동기화)

    단계:
    1. WAV 변환
    2. WhisperX (STT + 화자분리 + stt_cleanup 후처리)
    3. run_post_stt_pipeline (운율·화자·위험·트랜스크립트)
    """
    logger.info(f"[{processing_id}] v2 오디오 처리 시작: {input_path.name}")
    start_time = time.perf_counter()

    try:
        duration_seconds = get_audio_duration_seconds(input_path)

        if duration_seconds > settings.MAX_AUDIO_DURATION:
            raise ValueError(
                f"오디오가 너무 깁니다 ({duration_seconds:.0f}초). "
                f"최대 {settings.MAX_AUDIO_DURATION}초"
            )

        wav_path = settings.SHARED_CONVERTED_DIR / f"converted_{processing_id}.wav"
        convert_to_wav(input_path, wav_path)
        wav_path_str = str(wav_path)

        from app.utils.gcs import upload_to_gcs
        converted_gcs_path = f"audio/converted_{processing_id}.wav"
        audio_gcs_url = upload_to_gcs(wav_path, converted_gcs_path)

        stt_start = time.perf_counter()
        whisperx_result = transcribe_with_diarization(wav_path_str)
        stt_time = time.perf_counter() - stt_start
        segments = whisperx_result["segments"]
        logger.info(
            f"[{processing_id}] STT+화자분리 완료: {len(segments)}개 ({stt_time:.2f}초)"
        )

        pipeline_start = time.perf_counter()
        pipeline = run_post_stt_pipeline(wav_path_str, segments)
        pipeline_time = time.perf_counter() - pipeline_start
        logger.info(f"[{processing_id}] 운율·가드레일 완료 ({pipeline_time:.2f}초)")

        total_time = time.perf_counter() - start_time
        logger.info(f"[{processing_id}] v2 전체 처리 완료 ({total_time:.2f}초)")

        return {
            "success": True,
            "processing_id": processing_id,
            "duration_seconds": duration_seconds,
            "transcript": whisperx_result["transcript_plain"],
            "transcript_labeled": whisperx_result["transcript_labeled"],
            "transcript_enriched": pipeline["transcript_enriched"],
            "prosody_transcript": pipeline["prosody_transcript"],
            "analysis_meta": pipeline["analysis_meta"],
            "segments": pipeline["segments"],
            "speaker_roles": pipeline["speaker_roles"],
            "speaker_profiles": pipeline["speaker_profiles"],
            "emotion_data": pipeline["emotion_data"],
            "prosody_data": pipeline["prosody_data"],
            "segment_prosody": serialize_segment_prosody(pipeline["segment_prosody"]),
            "danger_detection": pipeline["danger_detection"],
            "wav_path": wav_path_str,
            "audio_gcs_url": audio_gcs_url,
            "timing": {
                "stt": round(stt_time, 2),
                "pipeline": round(pipeline_time, 2),
                "total": round(total_time, 2),
            },
        }

    except Exception as e:
        logger.error(f"[{processing_id}] v2 오디오 처리 실패: {e}", exc_info=True)
        return {
            "success": False,
            "processing_id": processing_id,
            "error": str(e),
        }
