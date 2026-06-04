"""
업그레이드된 오디오 처리 파이프라인 (v2)
WhisperX + 감정 + 프로소디 + 가드레일 + LLM 요약
"""
import time
import json
from pathlib import Path
from typing import Dict, Any
from celery import shared_task
from datetime import datetime, timedelta

from app.services.audio_handler_v2 import process_audio_file_v2
from app.services.audio_handler import cleanup_audio_files
from app.services.llm_summarizer_qwen8b_chunked_comma import summarize_transcript
from app.utils.time_utils import calculate_age
from app.utils.logger import get_logger
from app.utils.gcs import upload_to_gcs
from app.config import settings

logger = get_logger(__name__)


def save_result_to_file(processing_id: str, result: Dict[str, Any], **kwargs) -> None:
    """처리 결과를 JSON 파일로 저장"""
    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{file_timestamp}_{processing_id}.json"
        filepath = results_dir / filename

        if result.get("success") == False:
            save_data = {
                "success": False,
                "message": "오디오 처리 실패",
                "data": None,
                "error": result.get("error", "알 수 없는 오류")
            }
        else:
            save_data = {
                "success": True,
                "message": "오디오 처리 완료 (v2)",
                "data": {
                    "transcript": result.get("transcript", ""),
                    "transcript_labeled": result.get("transcript_labeled", ""),
                    "summary": result.get("summary", {}),
                    "speaker_roles": result.get("speaker_roles", {}),
                    "emotion_data": result.get("emotion_data", {}),
                    "prosody_data": result.get("prosody_data", {}),
                    "danger_detection": result.get("danger_detection", {}),
                    "processing_time": result.get("processing_time", 0),
                    "audio_duration": result.get("audio_duration", 0),
                    "timing": result.get("timing", {}),
                    "audio_gcs_path": result.get("audio_gcs_path"),
                    "input_gcs_path": result.get("input_gcs_path")
                }
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[{processing_id}] 결과 저장 완료: {filepath}")

    except Exception as e:
        logger.error(f"[{processing_id}] 결과 저장 실패: {e}")


@shared_task(name="app.tasks.audio_pipeline_v2.process_audio_task_v2", bind=True)
def process_audio_task_v2(
    self,
    file_path: str,
    filename: str,
    processing_id: str,
    admin_info: Dict[str, Any],
    user_info: Dict[str, Any],
    danger_info: Dict[str, Any],
    conversation_start_time: str
) -> Dict[str, Any]:
    """
    v2 오디오 처리 파이프라인
    STT(화자분리) → 감정 → 프로소디 → 가드레일 → LLM 요약
    """
    try:
        start_time = time.perf_counter()
        input_path = Path(file_path)

        # 0. 원본 GCS 업로드
        input_ext = input_path.suffix.lstrip(".")
        input_gcs_path = f"input/input_{processing_id}.{input_ext}"
        input_gcs_url = upload_to_gcs(input_path, input_gcs_path)

        # 1. v2 오디오 처리 (WhisperX + 감정 + 프로소디 + 가드레일)
        audio_result = process_audio_file_v2(input_path, processing_id)

        if not audio_result["success"]:
            raise RuntimeError(f"오디오 처리 실패: {audio_result['error']}")

        # 2. 시간 계산
        try:
            start_dt = datetime.strptime(conversation_start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = start_dt + timedelta(seconds=audio_result["duration_seconds"])
            start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = start_time_str

        # 3. LLM 요약 (enriched transcript + 감정/프로소디 메타데이터 전달)
        age = calculate_age(user_info.get("birth_date", ""))

        audio_analysis_meta = _build_analysis_meta(
            audio_result["speaker_roles"],
            audio_result["emotion_data"],
            audio_result["prosody_data"],
            audio_result["danger_detection"]
        )

        from app.services.llm_summarizer_qwen8b_chunked_comma import check_danger_detected
        danger_detected = check_danger_detected(danger_info)

        summary = summarize_transcript(
            transcript=audio_result["transcript_enriched"],
            start_time=start_time_str,
            end_time=end_time_str,
            duration_sec=round(audio_result["duration_seconds"]),
            admin_name=admin_info.get("name", ""),
            admin_phone=admin_info.get("phone", ""),
            patient_name=user_info.get("name", ""),
            age=age,
            gender=user_info.get("gender", ""),
            address=user_info.get("address", ""),
            danger_info=danger_info if danger_detected else None
        )

        # 감정/톤 정보 보강 (LLM 결과에 음성분석 결과 병합)
        summary = _enrich_summary_with_audio_analysis(
            summary, audio_result, audio_analysis_meta
        )

        if "위험요소" not in summary:
            summary["위험요소"] = danger_info

        # 4. 최종 결과
        total_time = time.perf_counter() - start_time
        result = {
            "processing_time": total_time,
            "audio_duration": audio_result["duration_seconds"],
            "summary": summary,
            "transcript": audio_result["transcript"],
            "transcript_labeled": audio_result["transcript_labeled"],
            "speaker_roles": audio_result["speaker_roles"],
            "emotion_data": audio_result["emotion_data"],
            "prosody_data": audio_result["prosody_data"],
            "danger_detection": audio_result["danger_detection"],
            "timing": audio_result["timing"],
            "audio_gcs_path": audio_result["audio_gcs_url"],
            "input_gcs_path": input_gcs_url
        }

        # 5. 파일 정리
        try:
            cleanup_audio_files(processing_id, input_path)
        except Exception as e:
            logger.warning(f"[{processing_id}] 파일 정리 실패: {e}")

        # 6. 결과 저장
        save_result_to_file(processing_id, result)
        logger.info(f"[{processing_id}] v2 파이프라인 완료 ({total_time:.2f}초)")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{processing_id}] v2 파이프라인 실패: {error_msg}")

        try:
            cleanup_audio_files(processing_id, Path(file_path))
        except:
            pass

        error_result = {"success": False, "error": error_msg}
        save_result_to_file(processing_id, error_result)
        raise RuntimeError(error_msg)


def _build_analysis_meta(
    speaker_roles: Dict,
    emotion_data: Dict,
    prosody_data: Dict,
    danger_detection: Dict
) -> str:
    """LLM 프롬프트에 추가할 음성분석 메타데이터 텍스트"""
    lines = ["[음성 분석 결과]"]

    for speaker, role in speaker_roles.items():
        emo = emotion_data.get(speaker, {})
        pros = prosody_data.get(speaker, {})
        tone = emo.get("tone_description", "분석 불가")
        voice = pros.get("voice_description", "분석 불가")
        lines.append(f"- {role}({speaker}): 감정={tone}, 음성={voice}")

    if danger_detection.get("detected"):
        lines.append(f"- ⚠ 위험 플래그: {', '.join(danger_detection['flags'])}")

    return "\n".join(lines)


def _enrich_summary_with_audio_analysis(
    summary: Dict,
    audio_result: Dict,
    analysis_meta: str
) -> Dict:
    """LLM 요약 결과에 음성분석 데이터 병합"""
    speaker_roles = audio_result.get("speaker_roles", {})
    emotion_data = audio_result.get("emotion_data", {})
    prosody_data = audio_result.get("prosody_data", {})

    manager_speaker = None
    senior_speaker = None
    for speaker, role in speaker_roles.items():
        if role == "매니저":
            manager_speaker = speaker
        elif role == "시니어":
            senior_speaker = speaker

    # 매니저 톤 보강
    if manager_speaker:
        emo = emotion_data.get(manager_speaker, {})
        pros = prosody_data.get(manager_speaker, {})
        if emo.get("tone_description"):
            summary["매니저의 대화 분위기, 톤"] = emo["tone_description"]
        if pros.get("voice_description"):
            summary["매니저의 대화 분위기, 톤"] += f" ({pros['voice_description']})"

    # 시니어 톤 보강
    if senior_speaker:
        emo = emotion_data.get(senior_speaker, {})
        pros = prosody_data.get(senior_speaker, {})
        if emo.get("tone_description"):
            summary["시니어의 대화 분위기, 톤"] = emo["tone_description"]
        if pros.get("voice_description"):
            summary["시니어의 대화 분위기, 톤"] += f" ({pros['voice_description']})"

    return summary
