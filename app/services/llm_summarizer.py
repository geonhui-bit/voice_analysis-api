"""
vLLM (Gemma-4-31B) 기반 보고서 생성
- .env의 LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS 사용
- OpenAI 호환 API (/v1/chat/completions)
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging
logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "report_prompt.txt"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:2000")
LLM_MODEL = os.getenv("LLM_MODEL", "/models/models/gemma-4-31B-it-AWQ-4bit")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))


def _load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _build_prompt(
    transcript: str,
    start_time: str,
    end_time: str,
    duration_sec: int,
    admin_name: str,
    admin_phone: str,
    patient_name: str,
    age: str,
    gender: str,
    address: str,
    analysis_meta: str = "",
    danger_info: Optional[Dict] = None,
) -> str:
    template = _load_prompt_template()

    prompt = template.format(
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
        transcript=transcript,
    )

    if danger_info:
        prompt += f"\n\n## 위험요소 사전 감지 정보\n{json.dumps(danger_info, ensure_ascii=False, indent=2)}"

    return prompt


def _call_llm(prompt: str) -> str:
    url = f"{LLM_BASE_URL}/v1/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.3,
    }

    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        logger.error(f"vLLM 응답 에러 ({response.status_code}): {response.text[:300]}")
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_json_response(text: str) -> Dict[str, Any]:
    import re

    # ```json ... ``` 코드블록 안의 JSON 추출
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 코드블록 없으면 첫 { ~ 마지막 } 사이 추출
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
    """
    가드레일 필터 출력 → LLM 보고서 생성

    Returns:
        보고서 JSON dict
    """
    try:
        prompt = _build_prompt(
            transcript=transcript,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            admin_name=admin_name,
            admin_phone=admin_phone,
            patient_name=patient_name,
            age=age,
            gender=gender,
            address=address,
            analysis_meta=analysis_meta,
            danger_info=danger_info,
        )

        MAX_PROMPT_CHARS = 50000
        if len(prompt) > MAX_PROMPT_CHARS:
            over = len(prompt) - MAX_PROMPT_CHARS
            logger.warning(f"프롬프트가 {len(prompt)}자로 너무 김, 대화 내용 {over}자 잘라냄")
            transcript_cutoff = len(transcript) - over - 200
            if transcript_cutoff > 500:
                prompt = _build_prompt(
                    transcript=transcript[:transcript_cutoff] + "\n... (이하 생략)",
                    start_time=start_time,
                    end_time=end_time,
                    duration_sec=duration_sec,
                    admin_name=admin_name,
                    admin_phone=admin_phone,
                    patient_name=patient_name,
                    age=age,
                    gender=gender,
                    address=address,
                    analysis_meta=analysis_meta,
                    danger_info=danger_info,
                )

        logger.info(f"LLM 호출 시작 (프롬프트 {len(prompt)}자)")
        raw_response = _call_llm(prompt)
        logger.info(f"LLM 응답 수신 ({len(raw_response)}자)")

        result = _parse_json_response(raw_response)
        return result

    except requests.exceptions.ConnectionError:
        logger.error(f"LLM 서버 연결 실패: {LLM_BASE_URL}")
        return {"error": "LLM 서버에 연결할 수 없습니다"}
    except requests.exceptions.Timeout:
        logger.error("LLM 응답 타임아웃 (120초)")
        return {"error": "LLM 응답 시간 초과"}
    except Exception as e:
        logger.error(f"LLM 요약 실패: {e}")
        return {"error": str(e)}


def check_danger_detected(danger_info: Optional[Dict]) -> bool:
    if not danger_info:
        return False
    return danger_info.get("detected", False)
