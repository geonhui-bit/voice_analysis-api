"""
음성 분석 API (GB10 Blackwell 호환)
Swagger UI: http://localhost:{API_PORT}/docs
"""
import os
import time
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.services.whisperx_loader import transcribe_with_diarization
from app.services.prosody_analyzer import analyze_prosody_full, format_prosody_inline_tag
from app.services.guard_filter import (
    identify_speaker_roles,
    detect_dangers_from_audio,
    build_enriched_transcript,
)
from app.services.transcript_builder import (
    build_analysis_meta,
    build_prosody_transcript,
    serialize_segment_prosody,
)
from app.services.llm_summarizer import (
    analyze_emotion_speakers,
    summarize_transcript,
    check_danger_detected,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="음성 분석 파이프라인 API",
    description=(
        "음성 → STT → 화자분리 → 운율 → 가드레일 → LLM. "
        "llm_mode: none(운율만) | emotion(감정·화자) | report(방문보고서)"
    ),
    version="0.2.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health_check():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {"status": "ok", "gpu": gpu}


@app.post("/analyze", summary="음성 파일 분석 (전체 파이프라인)")
async def analyze_audio(
    file: UploadFile = File(..., description="음성 파일 (wav, mp3, m4a 등)"),
    admin_name: str = Form("", description="매니저 이름"),
    admin_phone: str = Form("", description="매니저 연락처"),
    patient_name: str = Form("", description="시니어 이름"),
    age: str = Form("", description="시니어 나이"),
    gender: str = Form("", description="시니어 성별"),
    address: str = Form("", description="시니어 주소"),
    llm_mode: Literal["none", "emotion", "report"] = Form(
        "emotion",
        description="none=LLM없음, emotion=감정·화자분석(기본), report=방문보고서",
    ),
):
    """
    1. STT + 화자분리 → 2. 운율 → 3. 가드레일 → 4. LLM (mode에 따라)
    """
    processing_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    save_path = UPLOAD_DIR / f"{processing_id}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[{processing_id}] 파일 저장: {save_path}, llm_mode={llm_mode}")

    try:
        audio_path = str(save_path)
        timing = {}

        t0 = time.perf_counter()
        stt_result = transcribe_with_diarization(audio_path)
        timing["stt_diarization"] = round(time.perf_counter() - t0, 2)

        segments = stt_result["segments"]
        logger.info(
            f"[{processing_id}] STT+화자분리 완료: {len(segments)}개 ({timing['stt_diarization']}초)"
        )

        emotion_data = {}

        t0 = time.perf_counter()
        segment_prosody, prosody_data = analyze_prosody_full(audio_path, segments)
        timing["prosody"] = round(time.perf_counter() - t0, 2)
        logger.info(f"[{processing_id}] 운율분석 완료 ({timing['prosody']}초)")

        speaker_roles = identify_speaker_roles(segments, emotion_data, prosody_data)
        danger_detection = detect_dangers_from_audio(segments, emotion_data, prosody_data)
        enriched_transcript = build_enriched_transcript(segments, speaker_roles)

        duration_sec = int(segments[-1].get("end", 0)) if segments else 0
        analysis_meta = build_analysis_meta(speaker_roles, prosody_data, danger_detection)
        prosody_transcript = build_prosody_transcript(
            segments, segment_prosody, speaker_roles, format_prosody_inline_tag
        )

        emotion_analysis = None
        summary = None

        if llm_mode in ("emotion", "report"):
            t0 = time.perf_counter()
            danger_for_llm = (
                danger_detection if check_danger_detected(danger_detection) else None
            )

            if llm_mode == "emotion":
                emotion_analysis = analyze_emotion_speakers(
                    transcript=prosody_transcript,
                    analysis_meta=analysis_meta,
                    duration_sec=duration_sec,
                    danger_info=danger_for_llm,
                )
                timing["llm_emotion"] = round(time.perf_counter() - t0, 2)
                logger.info(
                    f"[{processing_id}] LLM 감정분석 완료 ({timing['llm_emotion']}초)"
                )

            if llm_mode == "report":
                now = datetime.now()
                start_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                end_time_str = (now + timedelta(seconds=duration_sec)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                summary = summarize_transcript(
                    transcript=prosody_transcript,
                    start_time=start_time_str,
                    end_time=end_time_str,
                    duration_sec=duration_sec,
                    admin_name=admin_name,
                    admin_phone=admin_phone,
                    patient_name=patient_name,
                    age=age,
                    gender=gender,
                    address=address,
                    analysis_meta=analysis_meta,
                    danger_info=danger_for_llm,
                )
                timing["llm_report"] = round(time.perf_counter() - t0, 2)
                logger.info(
                    f"[{processing_id}] LLM 보고서 완료 ({timing['llm_report']}초)"
                )

        timing["total"] = round(time.perf_counter() - start_time, 2)

        return {
            "processing_id": processing_id,
            "success": True,
            "llm_mode": llm_mode,
            "transcript_plain": stt_result["transcript_plain"],
            "transcript_labeled": stt_result["transcript_labeled"],
            "transcript_enriched": enriched_transcript,
            "prosody_transcript": prosody_transcript,
            "speaker_roles": speaker_roles,
            "emotion_data": emotion_data,
            "prosody_data": prosody_data,
            "segment_prosody": serialize_segment_prosody(segment_prosody),
            "danger_detection": danger_detection,
            "emotion_analysis": emotion_analysis,
            "summary": summary,
            "timing": timing,
        }

    except Exception as e:
        logger.error(f"[{processing_id}] 파이프라인 실패: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"processing_id": processing_id, "success": False, "error": str(e)},
        )

    finally:
        if save_path.exists():
            save_path.unlink()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "9002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
