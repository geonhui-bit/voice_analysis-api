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
from app.services.llm_summarizer import summarize_transcript, check_danger_detected

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="음성 분석 파이프라인 API",
    description="음성 파일 → STT → 화자분리 → 운율분석 → 가드레일 → LLM 보고서 (감정은 LLM이 텍스트+운율로 판단)",
    version="0.1.0",
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
    skip_llm: bool = Form(False, description="True면 LLM 보고서 생성 건너뜀"),
):
    """
    음성 파일을 업로드하면 전체 파이프라인을 실행합니다.

    1. STT (Whisper) → 2. 화자분리 (pyannote) → 3. 운율분석 (torchcrepe)
    → 4. 가드레일 → 5. LLM 보고서 (감정은 LLM이 텍스트+운율로 직접 판단)
    """
    processing_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    save_path = UPLOAD_DIR / f"{processing_id}_{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"[{processing_id}] 파일 저장: {save_path}")

    try:
        audio_path = str(save_path)
        timing = {}

        # 1~2. STT + 화자분리
        t0 = time.perf_counter()
        stt_result = transcribe_with_diarization(audio_path)
        timing["stt_diarization"] = round(time.perf_counter() - t0, 2)

        segments = stt_result["segments"]
        logger.info(f"[{processing_id}] STT+화자분리 완료: {len(segments)}개 세그먼트 ({timing['stt_diarization']}초)")

        # 3. 감정분석 (비활성화 - LLM이 텍스트+운율로 직접 판단)
        emotion_data = {}
        logger.info(f"[{processing_id}] 감정분석 스킵 (LLM 기반 감정 판단 사용)")

        # 4. 운율분석 (baseline + 상대값 + 급변, 1회 pass)
        t0 = time.perf_counter()
        segment_prosody, prosody_data = analyze_prosody_full(audio_path, segments)
        timing["prosody"] = round(time.perf_counter() - t0, 2)
        logger.info(f"[{processing_id}] 운율분석 완료 ({timing['prosody']}초)")

        # 5. 가드레일
        speaker_roles = identify_speaker_roles(segments, emotion_data, prosody_data)
        danger_detection = detect_dangers_from_audio(segments, emotion_data, prosody_data)
        enriched_transcript = build_enriched_transcript(segments, speaker_roles)

        # 6. LLM 보고서
        summary = None
        if not skip_llm:
            t0 = time.perf_counter()

            now = datetime.now()
            duration_sec = int(segments[-1].get("end", 0)) if segments else 0
            start_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = (now + timedelta(seconds=duration_sec)).strftime("%Y-%m-%d %H:%M:%S")

            # 화자별 종합 운율 (참고용)
            analysis_meta_lines = ["[화자별 운율 종합 (baseline 기준)]"]
            for speaker, role in speaker_roles.items():
                pros = prosody_data.get(speaker, {})
                voice_desc = pros.get("voice_description", "분석 불가")
                bp = pros.get("baseline_pitch", 0)
                analysis_meta_lines.append(
                    f"- {role}({speaker}): baseline {bp:.0f}Hz, {voice_desc}"
                )
            if danger_detection.get("detected"):
                analysis_meta_lines.append(f"- 위험 플래그: {', '.join(danger_detection['flags'])}")
            analysis_meta = "\n".join(analysis_meta_lines)

            # 세그먼트별 운율 인라인 트랜스크립트 생성
            prosody_transcript_lines = []
            for i, seg in enumerate(segments):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "").strip()
                role = speaker_roles.get(speaker, speaker)

                start_str = f"{int(start//60):02d}:{start%60:05.1f}"
                end_str = f"{int(end//60):02d}:{end%60:05.1f}"

                sp = segment_prosody[i] if i < len(segment_prosody) else None
                if sp and sp.get("prosody"):
                    pros_tag = format_prosody_inline_tag(sp["prosody"])
                    prosody_transcript_lines.append(
                        f"[{start_str}~{end_str}] [{role}] {text}  |운율: {pros_tag}|"
                    )
                else:
                    prosody_transcript_lines.append(
                        f"[{start_str}~{end_str}] [{role}] {text}"
                    )
            prosody_transcript = "\n".join(prosody_transcript_lines)

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
                danger_info=danger_detection if check_danger_detected(danger_detection) else None,
            )
            timing["llm"] = round(time.perf_counter() - t0, 2)
            logger.info(f"[{processing_id}] LLM 보고서 완료 ({timing['llm']}초)")

        total_time = round(time.perf_counter() - start_time, 2)
        timing["total"] = total_time

        return {
            "processing_id": processing_id,
            "success": True,
            "transcript_plain": stt_result["transcript_plain"],
            "transcript_labeled": stt_result["transcript_labeled"],
            "transcript_enriched": enriched_transcript,
            "speaker_roles": speaker_roles,
            "emotion_data": emotion_data,
            "prosody_data": prosody_data,
            "danger_detection": danger_detection,
            "summary": summary,
            "timing": timing,
        }

    except Exception as e:
        logger.error(f"[{processing_id}] 파이프라인 실패: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"processing_id": processing_id, "success": False, "error": str(e)}
        )

    finally:
        if save_path.exists():
            save_path.unlink()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "9002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
