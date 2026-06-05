"""
보고서 전 필터링 가드레일
- 화자 역할 자동 매핑 (매니저 vs 시니어)
- 이상 발화 감지
- 위험요소 사전 감지
"""
import os
import re
from collections import Counter
from typing import Dict, List, Any, Tuple
import logging
logger = logging.getLogger(__name__)

# 운율 급변 '다발' danger 플래그: 화자당 급변 횟수 하한 (기본 7, 실험 3→7)
PROSODY_SUDDEN_DANGER_MIN = int(os.getenv("PROSODY_DANGER_SUDDEN_MIN", "7"))

DIARIZATION_UNKNOWN = "UNKNOWN"
ROLE_UNASSIGNED = "미확인"
ROLE_SILENCE_GAP = "무음"

try:
    from app.services.stt_cleanup import SPEAKER_SILENCE_GAP
except ImportError:
    SPEAKER_SILENCE_GAP = "SILENCE_GAP"

MANAGER_KEYWORDS = [
    "어르신", "혈압", "재볼까", "드셨어요", "하셨어요",
    "다음에", "가져올게", "제가", "드릴게", "확인",
    "운동", "약", "식사", "가르쳐", "알려드릴게",
]

# 시니어 본인 발화(자기 건강·생활) — 매니저의 '어르신' 호칭과 구분
SENIOR_SELF_KEYWORDS = [
    "내가", "나는", "나도", "나한테", "내 ", "제 허리", "제 발",
    "아파", "아프", "아파서", "아파요", "통증", "병원", "입원", "간병",
    "먹었", "못 먹", "잠", "기력", "허리", "무릎", "식사", "고추",
    "며느리", "자식", "아들", "딸", "손주",
]

CO_RESIDENT_KEYWORDS = [
    "여보", "당신", "우리 남편", "우리 아내", "남편", "아내",
    "언니", "엄마하고", "아 엄마", "고추를", "따신", "잔소리",
]

# UNKNOWN(미할당) 세그먼트 역할 추론용
MANAGER_QUESTION_HINTS = (
    "드셨어", "하셨어", "드셔야", "안 드셨", "어떠셔", "맞지", "주사 맞",
    "갔다 오셨", "하셔서", "드시면", "먹어도 돼", "드려",
)
SENIOR_SHORT_REPLIES = (
    "괜찮아요", "괜찮아", "모르죠", "모르겠", "네", "응", "아이고", "음", "그래",
)
# UNKNOWN 추론용 (짧은 키워드 오매칭 방지 — '고추'≠시니어)
SENIOR_UNKNOWN_HINTS = (
    "내가", "나는", "나도", "아파", "아프", "병원", "허리", "통증", "먹었", "못 먹",
)

# 동거인/기타: 이 미만이면 시니어 후보 제외 → 동거인 쪽 배치
MIN_SPEAKER_SEGMENTS = int(os.getenv("MIN_SPEAKER_SEGMENTS", "3"))
MIN_SPEAKER_CHARS = int(os.getenv("MIN_SPEAKER_CHARS", "60"))


def role_for_speaker(speaker: str, speaker_roles: Dict[str, str]) -> str:
    """화자 ID → 표시 역할 (UNKNOWN은 speaker_roles·세그먼트 추론 참고)."""
    if speaker == SPEAKER_SILENCE_GAP:
        return ROLE_SILENCE_GAP
    if speaker == DIARIZATION_UNKNOWN:
        return speaker_roles.get(DIARIZATION_UNKNOWN, ROLE_UNASSIGNED)
    return speaker_roles.get(speaker, ROLE_UNASSIGNED)


def role_for_segment(seg: Dict, speaker_roles: Dict[str, str]) -> str:
    """세그먼트 단위 표시 역할 (UNKNOWN은 inferred_role 우선)."""
    if seg.get("is_silence_gap") or seg.get("speaker") == SPEAKER_SILENCE_GAP:
        return ROLE_SILENCE_GAP
    inferred = seg.get("inferred_role")
    if inferred:
        return inferred
    return role_for_speaker(seg.get("speaker", DIARIZATION_UNKNOWN), speaker_roles)

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


def _is_death_past_narrative(text: str) -> bool:
    """지인/가족 사망 회상·고인 언급 (돌아가+시/셨/신/었/다는 등)."""
    if any(marker in text for marker in _DEATH_PAST_NARRATIVE):
        return True
    return bool(
        re.search(
            r"돌아가(?:셨|신|었|시|실|심|셨다|신다|었다|다는|다는\s*거|다는\s*말)",
            text,
        )
    )


def _is_death_hypothetical(text: str) -> bool:
    """가정·과장 표현의 '죽었'."""
    if any(marker in text for marker in _DEATH_HYPOTHETICAL):
        return True
    return bool(re.search(r"(?:안\s*)?(?:했|하)으면\s*죽었|거의\s*죽었|안\s*죽었", text))


def is_danger_false_positive(danger_type: str, keyword: str, text: str) -> bool:
    """일상·회상·설명 맥락에서의 키워드 오탐 여부."""
    if danger_type == "사망":
        if keyword == "돌아가":
            return _is_death_past_narrative(text)
        if keyword == "죽었":
            return _is_death_hypothetical(text)
        if keyword == "사망":
            # '사망' 단독은 회상 문맥에서만 제외 (실제 위험 신호는 드묾)
            return _is_death_past_narrative(text) or _is_death_hypothetical(text)
    if danger_type == "통증":
        if "통풍" in text or any(ctx in text for ctx in _PAIN_EXPLANATION_CONTEXT):
            return True
    return False


def _collect_speaker_stats(segments: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """화자별 발화량·키워드 점수 집계."""
    stats: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        if seg.get("is_silence_gap") or seg.get("speaker") == SPEAKER_SILENCE_GAP:
            continue
        speaker = seg.get("speaker", DIARIZATION_UNKNOWN)
        if speaker == DIARIZATION_UNKNOWN:
            continue
        text = seg.get("text", "")
        if speaker not in stats:
            stats[speaker] = {
                "segments": 0,
                "chars": 0,
                "manager": 0.0,
                "senior": 0.0,
                "co_resident": 0.0,
                "questions": 0,
            }
        s = stats[speaker]
        s["segments"] += 1
        s["chars"] += len(text.replace(" ", ""))
        for kw in MANAGER_KEYWORDS:
            if kw in text:
                s["manager"] += 1
        for kw in SENIOR_SELF_KEYWORDS:
            if kw in text:
                s["senior"] += 1
        for kw in CO_RESIDENT_KEYWORDS:
            if kw in text:
                s["co_resident"] += 1
        if text.strip().endswith(("요?", "세요?", "까요?", "나요?", "죠?")):
            s["questions"] += 1
    return stats


def _is_active_speaker(stat: Dict[str, Any]) -> bool:
    return (
        stat["segments"] >= MIN_SPEAKER_SEGMENTS
        or stat["chars"] >= MIN_SPEAKER_CHARS
    )


def _manager_rank_score(stat: Dict[str, Any]) -> float:
    q_bonus = stat["questions"] / max(1, stat["segments"])
    return stat["manager"] + 0.5 * q_bonus


def _senior_rank_score(stat: Dict[str, Any]) -> float:
    if not _is_active_speaker(stat):
        return -1.0
    return stat["senior"] + 0.002 * stat["chars"] + 0.3 * stat["co_resident"]


def _infer_unknown_text_role(text: str, assigned_role_names: set) -> str:
    """pyannote 미할당(UNKNOWN) 발화 1건의 표시 역할 추론."""
    t = text.strip()
    if not t:
        return ROLE_UNASSIGNED

    m_score = sum(1 for kw in MANAGER_KEYWORDS if kw in t)
    for hint in MANAGER_QUESTION_HINTS:
        if hint in t:
            m_score += 2
    if t.endswith(("요?", "세요?", "까요?", "나요?", "죠?", "지?", "거야?")):
        m_score += 1

    s_score = sum(1 for kw in SENIOR_UNKNOWN_HINTS if kw in t)
    for short in SENIOR_SHORT_REPLIES:
        if t == short or t.startswith(short):
            s_score += 2
    if len(t) <= 6 and "?" not in t:
        s_score += 1

    c_score = sum(1 for kw in CO_RESIDENT_KEYWORDS if kw in t)
    if t.endswith("?") and c_score > 0 and m_score == 0:
        c_score += 1

    scores = {"매니저": m_score, "시니어": s_score, "동거인": c_score}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        if {"매니저", "시니어"}.issubset(assigned_role_names):
            return "동거인"
        return ROLE_UNASSIGNED
    return best


def _annotate_unknown_segments(
    segments: List[Dict],
    speaker_roles: Dict[str, str],
) -> None:
    """UNKNOWN 세그먼트에 inferred_role 부여, speaker_roles[UNKNOWN] 정리."""
    assigned_names = set(speaker_roles.values())
    inferred_roles: List[str] = []

    for seg in segments:
        if seg.get("is_silence_gap") or seg.get("speaker") != DIARIZATION_UNKNOWN:
            continue
        role = _infer_unknown_text_role(seg.get("text", ""), assigned_names)
        seg["inferred_role"] = role
        if role != ROLE_UNASSIGNED:
            inferred_roles.append(role)

    if not any(
        s.get("speaker") == DIARIZATION_UNKNOWN and not s.get("is_silence_gap")
        for s in segments
    ):
        return

    if {"매니저", "시니어"}.issubset(assigned_names):
        if inferred_roles:
            top_role, _ = Counter(inferred_roles).most_common(1)[0]
            speaker_roles[DIARIZATION_UNKNOWN] = top_role
        else:
            speaker_roles[DIARIZATION_UNKNOWN] = "동거인"
        logger.info(
            f"[화자] UNKNOWN → {speaker_roles[DIARIZATION_UNKNOWN]} "
            f"({len(inferred_roles)}건 세그먼트 추론)"
        )
    elif DIARIZATION_UNKNOWN not in speaker_roles:
        speaker_roles[DIARIZATION_UNKNOWN] = ROLE_UNASSIGNED


def compute_speaker_profiles(
    segments: List[Dict],
    speaker_roles: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """API·LLM 메타: 발화 수, sparse 여부."""
    stats = _collect_speaker_stats(segments)
    profiles: Dict[str, Dict[str, Any]] = {}
    for speaker, s in stats.items():
        sparse = not _is_active_speaker(s)
        profiles[speaker] = {
            "role": speaker_roles.get(speaker, ROLE_UNASSIGNED),
            "segment_count": s["segments"],
            "char_count": s["chars"],
            "sparse": sparse,
        }

    unknown_segs = [
        s
        for s in segments
        if s.get("speaker") == DIARIZATION_UNKNOWN and not s.get("is_silence_gap")
    ]
    if unknown_segs:
        profiles[DIARIZATION_UNKNOWN] = {
            "role": speaker_roles.get(DIARIZATION_UNKNOWN, ROLE_UNASSIGNED),
            "segment_count": len(unknown_segs),
            "char_count": sum(len(s.get("text", "").replace(" ", "")) for s in unknown_segs),
            "sparse": True,
        }
    return profiles


def identify_speaker_roles(
    segments: List[Dict],
    emotion_data: Dict = None,
    prosody_data: Dict = None
) -> Dict[str, str]:
    """
    화자 역할 자동 매핑 (3인 이상 대화)

    1. 매니저: 매니저 키워드·질문 비율 최다
    2. 시니어: 남은 화자 중 시니어 자기서술 키워드 + 발화량 충분
    3. 동거인 / 동거인_2: 발화 적은 3번째 화자 등 (시니어로 올리지 않음)
    """
    stats = _collect_speaker_stats(segments)
    has_unknown = any(
        seg.get("speaker") == DIARIZATION_UNKNOWN
        for seg in segments
        if not seg.get("is_silence_gap")
    )

    roles: Dict[str, str] = {}
    if not stats:
        if has_unknown:
            _annotate_unknown_segments(segments, roles)
        return roles

    manager_sp = max(stats.keys(), key=lambda sp: _manager_rank_score(stats[sp]))
    roles[manager_sp] = "매니저"
    rest = [sp for sp in stats if sp != manager_sp]

    senior_candidates = [sp for sp in rest if _senior_rank_score(stats[sp]) >= 0]
    if senior_candidates:
        senior_sp = max(senior_candidates, key=lambda sp: _senior_rank_score(stats[sp]))
    elif rest:
        senior_sp = max(rest, key=lambda sp: stats[sp]["chars"])
    else:
        senior_sp = None

    if senior_sp:
        roles[senior_sp] = "시니어"
        rest = [sp for sp in rest if sp != senior_sp]

    co_idx = 1
    for sp in sorted(rest, key=lambda s: stats[s]["chars"], reverse=True):
        roles[sp] = "동거인" if co_idx == 1 else f"동거인_{co_idx}"
        co_idx += 1

    _annotate_unknown_segments(segments, roles)

    profiles = compute_speaker_profiles(segments, roles)
    sparse_info = [
        f"{profiles[sp]['role']}({sp}:{profiles[sp]['segment_count']}회)"
        for sp in profiles
        if profiles[sp]["sparse"]
    ]
    if sparse_info:
        logger.info(f"[화자] 발화 부족: {', '.join(sparse_info)}")
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
    keyword_flags: List[str] = []
    prosody_flags: List[str] = []
    emotion_flags: List[str] = []
    details = []
    details_added_for_seg: set = set()

    # 1. 텍스트 기반 위험 키워드 — 세그먼트·타입당 1건만 (아파/아프 중복 방지)
    for seg in segments:
        if seg.get("is_silence_gap") or seg.get("speaker") == SPEAKER_SILENCE_GAP:
            continue
        text = seg.get("text", "")
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        seg_key = (speaker, round(float(start), 1))

        for danger_type, keywords in DANGER_KEYWORDS.items():
            if (danger_type, seg_key) in details_added_for_seg:
                continue
            for kw in keywords:
                if kw not in text:
                    continue
                if is_danger_false_positive(danger_type, kw, text):
                    logger.debug(
                        f"[위험필터] 문맥 제외 ({danger_type}/{kw}): {text[:50]}..."
                    )
                    continue
                details.append({
                    "type": danger_type,
                    "keyword": kw,
                    "speaker": speaker,
                    "time": start,
                    "text": text,
                })
                details_added_for_seg.add((danger_type, seg_key))
                break

    keyword_types = {d["type"] for d in details}
    keyword_flags = [f"{t} 키워드 감지" for t in sorted(keyword_types)]

    # 2. 감정 기반 위험 감지
    if emotion_data:
        for speaker, emo in emotion_data.items():
            if emo.get("dominant_emotion") == "angry":
                dist = emo.get("emotion_distribution", {})
                if dist.get("angry", 0) > 0.4:
                    emotion_flags.append(f"{speaker} 강한 분노 감정 감지")

            if emo.get("dominant_emotion") == "sad":
                dist = emo.get("emotion_distribution", {})
                if dist.get("sad", 0) > 0.5:
                    emotion_flags.append(f"{speaker} 심한 우울/슬픔 감지")

    # 3. 프로소디 기반 위험 감지 (고성/급변)
    if prosody_data:
        for speaker, pros in prosody_data.items():
            if speaker == DIARIZATION_UNKNOWN:
                continue
            rel = pros.get("dominant_relative_pitch", "normal")
            if pros.get("energy_level") == "high" and rel in ("higher", "much_higher"):
                prosody_flags.append(f"{speaker} 고성 가능성")
            sudden_n = pros.get("sudden_change_count", 0)
            if sudden_n >= PROSODY_SUDDEN_DANGER_MIN:
                prosody_flags.append(
                    f"{speaker} 운율 급변 다발 ({sudden_n}회, 기준>={PROSODY_SUDDEN_DANGER_MIN})"
                )

    flags = keyword_flags + emotion_flags + prosody_flags
    if keyword_flags:
        logger.info(f"[위험필터] 키워드 flags: {keyword_flags} (details {len(details)}건)")

    return {
        "detected": len(flags) > 0,
        "flags": flags,
        "details": details,
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
        text = seg.get("text", "").strip()
        role = role_for_segment(seg, speaker_roles)

        start_str = f"{int(start//60):02d}:{start%60:06.3f}"
        end_str = f"{int(end//60):02d}:{end%60:06.3f}"
        lines.append(f"[{start_str} --> {end_str}] [{role}] {text}")

    return "\n".join(lines)
