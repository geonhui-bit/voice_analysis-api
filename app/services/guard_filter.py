"""
보고서 전 필터링 가드레일
- 화자 역할 자동 매핑 (매니저 vs 시니어)
- 이상 발화 감지
- 위험요소 사전 감지
"""
import os
from typing import Dict, List, Any, Tuple
import logging
logger = logging.getLogger(__name__)

# 운율 급변 '다발' danger 플래그: 화자당 급변 횟수 하한 (기본 7, 실험 3→7)
PROSODY_SUDDEN_DANGER_MIN = int(os.getenv("PROSODY_DANGER_SUDDEN_MIN", "7"))

DIARIZATION_UNKNOWN = "UNKNOWN"
ROLE_UNASSIGNED = "미확인"

MANAGER_KEYWORDS = [
    "어르신", "혈압", "재볼까", "드셨어요", "하셨어요",
    "다음에", "가져올게", "제가", "드릴게", "확인",
    "운동", "약", "식사", "가르쳐", "알려드릴게"
]


def role_for_speaker(speaker: str, speaker_roles: Dict[str, str]) -> str:
    """화자 ID → 표시 역할 (미매핑·UNKNOWN은 미확인)."""
    if speaker == DIARIZATION_UNKNOWN:
        return ROLE_UNASSIGNED
    return speaker_roles.get(speaker, ROLE_UNASSIGNED)

DANGER_KEYWORDS = {
    "폭언": ["씨발", "개새끼", "죽여", "꺼져"],
    "낙상": ["넘어졌", "떨어졌", "미끄러", "넘어질"],
    "사망": ["돌아가", "죽었", "사망"],
    "통증": ["아파", "아프", "쑤시", "찌릿"],
}

# 사망 키워드: 과거 회상·가족 사망 이야기 (실시간 위험 아님)
_DEATH_PAST_NARRATIVE = (
    "돌아가셨",
    "돌아가신",
    "돌아가셨지만",
    "돌아가셨다",
    "돌아가셨다는",
    "돌아가셨군",
    "돌아가셨어",
    "돌아가신",
    "돌아간 때",
    "돌아가신",
)

# 사망 키워드: 가정법·과장 표현
_DEATH_HYPOTHETICAL = (
    "안했으면 죽었",
    "안 했으면 죽었",
    "않았으면 죽었",
    "않으면 죽었",
    "안하면 죽었",
    "하면 죽었",
    "거의 죽었",
)

# 통증 키워드: 질병 설명·상담 맥락 (응급 호출 신호로 보기 어려움)
_PAIN_EXPLANATION_CONTEXT = (
    "통풍이 그래요",
    "왜 통풍인가",
    "청풍이란",
    "바람만 스쳐도",
    "그 정도로 아파요",
    "너무 아프죠",
    "통증이 온다",
    "병원에서",
    "주사를",
    "물을 뺐",
)


def is_danger_false_positive(danger_type: str, keyword: str, text: str) -> bool:
    """일상·회상·설명 맥락에서의 키워드 오탐 여부."""
    if danger_type == "사망":
        if keyword == "돌아가":
            return any(marker in text for marker in _DEATH_PAST_NARRATIVE)
        if keyword == "죽었":
            return any(marker in text for marker in _DEATH_HYPOTHETICAL)
    if danger_type == "통증":
        if "통풍" in text or any(ctx in text for ctx in _PAIN_EXPLANATION_CONTEXT):
            return True
    return False


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

    has_unknown = False
    for seg in segments:
        speaker = seg.get("speaker", DIARIZATION_UNKNOWN)
        if speaker == DIARIZATION_UNKNOWN:
            has_unknown = True
            continue
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

    roles: Dict[str, str] = {}
    if speaker_scores:
        sorted_speakers = sorted(speaker_scores.items(), key=lambda x: x[1], reverse=True)
        roles[sorted_speakers[0][0]] = "매니저"
        for i, (speaker, _score) in enumerate(sorted_speakers[1:], 1):
            if i == 1:
                roles[speaker] = "시니어"
            else:
                roles[speaker] = f"동거인_{i}"

    if has_unknown:
        roles[DIARIZATION_UNKNOWN] = ROLE_UNASSIGNED

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
                if kw not in text:
                    continue
                if is_danger_false_positive(danger_type, kw, text):
                    logger.debug(
                        f"[위험필터] 문맥 제외 ({danger_type}/{kw}): {text[:50]}..."
                    )
                    continue
                flags.append(f"{danger_type} 키워드 감지")
                details.append({
                    "type": danger_type,
                    "keyword": kw,
                    "speaker": speaker,
                    "time": start,
                    "text": text,
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
            if speaker == DIARIZATION_UNKNOWN:
                continue
            rel = pros.get("dominant_relative_pitch", "normal")
            if pros.get("energy_level") == "high" and rel in ("higher", "much_higher"):
                flags.append(f"{speaker} 고성 가능성")
            sudden_n = pros.get("sudden_change_count", 0)
            if sudden_n >= PROSODY_SUDDEN_DANGER_MIN:
                flags.append(
                    f"{speaker} 운율 급변 다발 ({sudden_n}회, 기준>={PROSODY_SUDDEN_DANGER_MIN})"
                )

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
        role = role_for_speaker(speaker, speaker_roles)

        start_str = f"{int(start//60):02d}:{start%60:06.3f}"
        end_str = f"{int(end//60):02d}:{end%60:06.3f}"
        lines.append(f"[{start_str} --> {end_str}] [{role}] {text}")

    return "\n".join(lines)
