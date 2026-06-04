"""
vLLM (Gemma-4-31B) 기반 LLM 호출
- emotion_prompt.txt: 화자별 감정·톤 분석
- report_prompt.txt: 방문 요양 보고서 (추후/선택)
"""
import os
import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

REPORT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "report_prompt.txt"
EMOTION_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "emotion_prompt.txt"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:2000")
LLM_MODEL = os.getenv("LLM_MODEL", "/models/models/gemma-4-31B-it-AWQ-4bit")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_EMOTION_MAX_TOKENS = int(os.getenv("LLM_EMOTION_MAX_TOKENS", "2048"))

MAX_PROMPT_CHARS = 50000


def _load_prompt_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _call_llm(prompt: str, max_tokens: int) -> str:
    url = f"{LLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    response = requests.post(url, json=payload, timeout=180)
    if response.status_code != 200:
        logger.error(f"vLLM 응답 에러 ({response.status_code}): {response.text[:300]}")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _parse_json_response(text: str) -> Dict[str, Any]:
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("LLM 응답에서 JSON을 찾을 수 없음, 원문 반환")
        return {"raw_response": text}

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패: {e}")
        return {"raw_response": text}


def _truncate_transcript_in_prompt(
    prompt: str,
    transcript: str,
    rebuild_fn,
) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    over = len(prompt) - MAX_PROMPT_CHARS
    logger.warning(f"프롬프트가 {len(prompt)}자로 너무 김, 대화 내용 {over}자 잘라냄")
    cutoff = len(transcript) - over - 200
    if cutoff > 500:
        return rebuild_fn(transcript[:cutoff] + "\n... (이하 생략)")
    return prompt


def analyze_emotion_speakers(
    transcript: str,
    analysis_meta: str,
    duration_sec: int = 0,
    danger_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """텍스트 + 운율 → 화자별 감정·톤 분석 (보고서 없음)"""
    try:
        template = _load_prompt_template(EMOTION_PROMPT_PATH)

        def build_prompt(tr: str) -> str:
            p = template.format(
                duration_sec=duration_sec,
                analysis_meta=analysis_meta or "분석 데이터 없음",
                transcript=tr,
            )
            if danger_info:
                p += f"\n\n## 위험요소 사전 감지 정보\n{json.dumps(danger_info, ensure_ascii=False, indent=2)}"
            return p

        prompt = build_prompt(transcript)
        prompt = _truncate_transcript_in_prompt(prompt, transcript, build_prompt)

        logger.info(f"LLM 감정분석 호출 (프롬프트 {len(prompt)}자)")
        raw = _call_llm(prompt, LLM_EMOTION_MAX_TOKENS)
        logger.info(f"LLM 감정분석 응답 ({len(raw)}자)")
        return _parse_json_response(raw)

    except requests.exceptions.ConnectionError:
        logger.error(f"LLM 서버 연결 실패: {LLM_BASE_URL}")
        return {"error": "LLM 서버에 연결할 수 없습니다"}
    except requests.exceptions.Timeout:
        logger.error("LLM 응답 타임아웃")
        return {"error": "LLM 응답 시간 초과"}
    except Exception as e:
        logger.error(f"LLM 감정분석 실패: {e}")
        return {"error": str(e)}


def summarize_transcript(
    transcript: str,
    start_time: str,
    end_time: str,
    duration_sec: int,
    admin_name: str = "",
    admin_phone: str = "",
    patient_name: str = "",
    age: str = "",
    gender: str = "",
    address: str = "",
    analysis_meta: str = "",
    danger_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """방문 요양 보고서 JSON 생성 (llm_mode=report)"""
    try:
        template = _load_prompt_template(REPORT_PROMPT_PATH)

        def build_prompt(tr: str) -> str:
            p = template.format(
                admin_name=admin_name or "미확인",
                admin_phone=admin_phone or "미확인",
                patient_name=patient_name or "미확인",
                age=age or "미확인",
                gender=gender or "미확인",
                address=address or "미확인",
                start_time=start_time,
                end_time=end_time,
                duration_sec=duration_sec,
                analysis_meta=analysis_meta or "분석 데이터 없음",
                transcript=tr,
            )
            if danger_info:
                p += f"\n\n## 위험요소 사전 감지 정보\n{json.dumps(danger_info, ensure_ascii=False, indent=2)}"
            return p

        prompt = build_prompt(transcript)
        prompt = _truncate_transcript_in_prompt(prompt, transcript, build_prompt)

        logger.info(f"LLM 보고서 호출 (프롬프트 {len(prompt)}자)")
        raw = _call_llm(prompt, LLM_MAX_TOKENS)
        logger.info(f"LLM 보고서 응답 ({len(raw)}자)")
        return _parse_json_response(raw)

    except requests.exceptions.ConnectionError:
        logger.error(f"LLM 서버 연결 실패: {LLM_BASE_URL}")
        return {"error": "LLM 서버에 연결할 수 없습니다"}
    except requests.exceptions.Timeout:
        logger.error("LLM 응답 타임아웃")
        return {"error": "LLM 응답 시간 초과"}
    except Exception as e:
        logger.error(f"LLM 보고서 실패: {e}")
        return {"error": str(e)}


def check_danger_detected(danger_info: Optional[Dict]) -> bool:
    if not danger_info:
        return False
    return danger_info.get("detected", False)
