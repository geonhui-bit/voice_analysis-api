"""
보고서 전 필터링 가드레일
- 화자 역할 자동 매핑 (매니저 vs 시니어)
- 이상 발화 감지
- 위험요소 사전 감지
"""
from typing import Dict, List, Any, Tuple
import logging
logger = logging.getLogger(__name__)

MANAGER_KEYWORDS = [
    "어르신", "혈압", "재볼까", "드셨어요", "하셨어요",
    "다음에", "가져올게", "제가", "드릴게", "확인",
    "운동", "약", "식사", "가르쳐", "알려드릴게"
]

DANGER_KEYWORDS = {
    "폭언": ["씨발", "개새끼", "죽여", "꺼져"],
    "낙상": ["넘어졌", "떨어졌", "미끄러", "넘어질"],
    "사망": ["돌아가", "죽었", "사망"],
    "통증": ["아파", "아프", "쑤시", "찌릿"]
}


def identify_speaker_roles(
    segments: List[Dict],
    emotion_data: Dict = None,
    prosody_data: Dict = None
) -> Dict[str, str]:
    """
    화자 역할 자동 매핑

    규칙:
    1. 매니저 키워드를 더 많이 사용하는 화자 = 매니저
    2. 첫 발화가 인사/소개인 화자 = 매니저
    3. 질문(~요?, ~세요?) 비율 높은 화자 = 매니저

    Returns:
        {"SPEAKER_00": "매니저", "SPEAKER_01": "시니어", "SPEAKER_02": "동거인"}
    """
    speaker_scores = {}
    speaker_texts = {}

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")

        if speaker not in speaker_scores:
            speaker_scores[speaker] = 0
            speaker_texts[speaker] = []

        speaker_texts[speaker].append(text)

        for keyword in MANAGER_KEYWORDS:
            if keyword in text:
                speaker_scores[speaker] += 1

        if text.strip().endswith(("요?", "세요?", "까요?", "나요?", "죠?")):
            speaker_scores[speaker] += 0.5

    if not speaker_scores:
        return {}

    sorted_speakers = sorted(speaker_scores.items(), key=lambda x: x[1], reverse=True)

    roles = {}
    roles[sorted_speakers[0][0]] = "매니저"

    for i, (speaker, score) in enumerate(sorted_speakers[1:], 1):
        if i == 1:
            roles[speaker] = "시니어"
        else:
            roles[speaker] = f"동거인_{i}"

    logger.info(f"화자 역할 매핑: {roles}")
    return roles


def detect_dangers_from_audio(
    segments: List[Dict],
    emotion_data: Dict = None,
    prosody_data: Dict = None
) -> Dict[str, Any]:
    """
    음성 데이터 기반 위험요소 사전 감지

    Returns:
        {
            "detected": True/False,
            "flags": ["고성 감지", "부정 감정 지속"],
            "details": [...]
        }
    """
    flags = []
    details = []

    # 1. 텍스트 기반 위험 키워드 감지
    for seg in segments:
        text = seg.get("text", "")
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)

        for danger_type, keywords in DANGER_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    flags.append(f"{danger_type} 키워드 감지")
                    details.append({
                        "type": danger_type,
                        "keyword": kw,
                        "speaker": speaker,
                        "time": start,
                        "text": text
                    })

    # 2. 감정 기반 위험 감지
    if emotion_data:
        for speaker, emo in emotion_data.items():
            if emo.get("dominant_emotion") == "angry":
                dist = emo.get("emotion_distribution", {})
                if dist.get("angry", 0) > 0.4:
                    flags.append(f"{speaker} 강한 분노 감정 감지")

            if emo.get("dominant_emotion") == "sad":
                dist = emo.get("emotion_distribution", {})
                if dist.get("sad", 0) > 0.5:
                    flags.append(f"{speaker} 심한 우울/슬픔 감지")

    # 3. 프로소디 기반 위험 감지 (고성/급변)
    if prosody_data:
        for speaker, pros in prosody_data.items():
            rel = pros.get("dominant_relative_pitch", "normal")
            if pros.get("energy_level") == "high" and rel in ("higher", "much_higher"):
                flags.append(f"{speaker} 고성 가능성")
            if pros.get("sudden_change_count", 0) >= 3:
                flags.append(f"{speaker} 운율 급변 다발 ({pros['sudden_change_count']}회)")

    return {
        "detected": len(flags) > 0,
        "flags": list(set(flags)),
        "details": details
    }


def build_enriched_transcript(
    segments: List[Dict],
    speaker_roles: Dict[str, str]
) -> str:
    """LLM에 넘길 화자 역할 포함 트랜스크립트 생성"""
    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        role = speaker_roles.get(speaker, speaker)

        start_str = f"{int(start//60):02d}:{start%60:06.3f}"
        end_str = f"{int(end//60):02d}:{end%60:06.3f}"
        lines.append(f"[{start_str} --> {end_str}] [{role}] {text}")

    return "\n".join(lines)
