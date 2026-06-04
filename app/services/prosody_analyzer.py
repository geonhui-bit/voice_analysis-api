"""
torchcrepe + librosa 기반 피치/에너지/말속도 분석 (GB10 Blackwell 호환)
- 피치: torchcrepe (PyTorch GPU)
- 에너지: librosa RMS (CPU)
- 말속도: librosa onset detection (CPU)
- 화자별 baseline 대비 상대값 + pitch 변동 + 급변 감지
"""
import torch
import numpy as np
import librosa
import torchcrepe
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

import logging
logger = logging.getLogger(__name__)

MIN_SEGMENT_SEC = 1.5
SUDDEN_PITCH_RATIO = 0.25
SUDDEN_ENERGY_RATIO = 0.40
ROLLING_WINDOW = 3


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def analyze_prosody_for_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    sr: int = 16000
) -> Dict[str, Any]:
    """특정 구간의 프로소디(피치, 에너지, 말속도) 분석"""
    try:
        y, _ = librosa.load(
            audio_path, sr=sr, offset=start_sec, duration=end_sec - start_sec
        )

        if len(y) < sr * 0.5:
            return _empty_prosody()

        device = get_device()

        audio_tensor = torch.tensor(y).unsqueeze(0).to(device)
        frequency = torchcrepe.predict(
            audio_tensor, sr,
            hop_length=160,
            fmin=50, fmax=500,
            model='tiny',
            decoder=torchcrepe.decode.viterbi,
            device=device,
            return_periodicity=False,
            batch_size=1024
        )

        freq_np = frequency.cpu().numpy().flatten()
        valid_freq = freq_np[(freq_np > 50) & (freq_np < 500)]

        if len(valid_freq) > 0:
            pitch_mean = float(np.mean(valid_freq))
            pitch_std = float(np.std(valid_freq))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0

        if pitch_mean < 100:
            pitch_level = "low"
        elif pitch_mean > 250:
            pitch_level = "high"
        else:
            pitch_level = "normal"

        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))

        if energy_mean < 0.01:
            energy_level = "low"
        elif energy_mean > 0.08:
            energy_level = "high"
        else:
            energy_level = "normal"

        duration = end_sec - start_sec
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        syllables_per_sec = len(onsets) / duration if duration > 0 else 0

        if syllables_per_sec < 3:
            speech_rate = "slow"
        elif syllables_per_sec > 7:
            speech_rate = "fast"
        else:
            speech_rate = "normal"

        return {
            "pitch_mean": round(pitch_mean, 1),
            "pitch_std": round(pitch_std, 1),
            "pitch_level": pitch_level,
            "energy_mean": round(energy_mean, 4),
            "energy_level": energy_level,
            "speech_rate": speech_rate
        }

    except Exception as e:
        logger.warning(f"프로소디 분석 실패 ({start_sec:.1f}~{end_sec:.1f}s): {e}")
        return _empty_prosody()


def _empty_prosody() -> Dict[str, Any]:
    return {
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "pitch_level": "unknown",
        "energy_mean": 0.0,
        "energy_level": "unknown",
        "speech_rate": "unknown"
    }


def _is_valid_prosody(prosody: Optional[Dict[str, Any]]) -> bool:
    return bool(prosody and prosody.get("pitch_level") != "unknown")


def _compute_speaker_baselines(
    segment_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """화자별 baseline (pitch/energy/std 중앙값)"""
    by_speaker: Dict[str, List[Dict]] = {}
    for item in segment_results:
        if not _is_valid_prosody(item.get("prosody")):
            continue
        speaker = item["speaker"]
        by_speaker.setdefault(speaker, []).append(item["prosody"])

    baselines = {}
    for speaker, prosodies in by_speaker.items():
        baselines[speaker] = {
            "baseline_pitch": float(np.median([p["pitch_mean"] for p in prosodies])),
            "baseline_energy": float(np.median([p["energy_mean"] for p in prosodies])),
            "baseline_pitch_std": float(np.median([p["pitch_std"] for p in prosodies])),
        }
    return baselines


def _pitch_var_level(pitch_std: float, baseline_pitch_std: float) -> str:
    if pitch_std >= max(baseline_pitch_std * 1.5, 50):
        return "high"
    if pitch_std <= min(baseline_pitch_std * 0.8, 25):
        return "low"
    return "normal"


def _relative_pitch_label(ratio: float) -> str:
    if ratio < 0.85:
        return "much_lower"
    if ratio < 0.95:
        return "lower"
    if ratio <= 1.05:
        return "normal"
    if ratio <= 1.15:
        return "higher"
    return "much_higher"


_REL_PITCH_KO = {
    "much_lower": "훨씬낮음",
    "lower": "낮음",
    "normal": "평소",
    "higher": "높음",
    "much_higher": "훨씬높음",
}

_VAR_KO = {"low": "안정", "normal": "보통", "high": "큼"}


def _detect_sudden_change(
    prosody: Dict[str, Any],
    recent: List[Dict[str, Any]],
    baseline: Dict[str, float],
) -> Tuple[bool, str]:
    if not recent:
        ref_pitch = baseline["baseline_pitch"]
        ref_energy = baseline["baseline_energy"]
    else:
        ref_pitch = float(np.mean([p["pitch_mean"] for p in recent]))
        ref_energy = float(np.mean([p["energy_mean"] for p in recent]))

    pitch_delta = abs(prosody["pitch_mean"] - ref_pitch) / ref_pitch if ref_pitch > 0 else 0
    energy_delta = abs(prosody["energy_mean"] - ref_energy) / ref_energy if ref_energy > 0 else 0

    hints = []
    if pitch_delta >= SUDDEN_PITCH_RATIO:
        hints.append("피치↑" if prosody["pitch_mean"] > ref_pitch else "피치↓")
    if energy_delta >= SUDDEN_ENERGY_RATIO:
        hints.append("에너지↑" if prosody["energy_mean"] > ref_energy else "에너지↓")

    if hints:
        return True, "+".join(hints)
    return False, ""


def _enrich_prosody(
    prosody: Dict[str, Any],
    baseline: Dict[str, float],
    recent_same_speaker: List[Dict[str, Any]],
) -> Dict[str, Any]:
    enriched = dict(prosody)
    bp = baseline["baseline_pitch"]
    be = baseline["baseline_energy"]
    bps = baseline["baseline_pitch_std"]

    pitch_ratio = prosody["pitch_mean"] / bp if bp > 0 else 1.0
    energy_ratio = prosody["energy_mean"] / be if be > 0 else 1.0

    sudden, change_hint = _detect_sudden_change(prosody, recent_same_speaker, baseline)

    enriched.update({
        "baseline_pitch": round(bp, 1),
        "baseline_energy": round(be, 4),
        "pitch_vs_baseline_pct": round((pitch_ratio - 1.0) * 100, 1),
        "energy_vs_baseline_pct": round((energy_ratio - 1.0) * 100, 1),
        "relative_pitch": _relative_pitch_label(pitch_ratio),
        "pitch_var_level": _pitch_var_level(prosody["pitch_std"], bps),
        "sudden_change": sudden,
        "change_hint": change_hint,
    })
    return enriched


def format_prosody_inline_tag(prosody: Dict[str, Any]) -> str:
    """LLM 인라인 운율 태그 문자열"""
    rel = _REL_PITCH_KO.get(prosody.get("relative_pitch", "normal"), "평소")
    var = _VAR_KO.get(prosody.get("pitch_var_level", "normal"), "보통")
    parts = [
        f"피치={prosody['pitch_mean']:.0f}Hz({rel},{prosody['pitch_vs_baseline_pct']:+.0f}%)",
        f"변동={var}(std={prosody['pitch_std']:.0f})",
        f"에너지={prosody['energy_level']}({prosody['energy_vs_baseline_pct']:+.0f}%)",
        f"말속도={prosody['speech_rate']}",
    ]
    if prosody.get("sudden_change"):
        parts.append(f"급변={prosody['change_hint']}")
    return ", ".join(parts)


def analyze_prosody_full(
    audio_path: str,
    segments: List[Dict]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    세그먼트별 운율 분석 + 화자 baseline/상대값/급변 → 한 번에 수행

    Returns:
        (segment_prosody_list, speaker_prosody_dict)
    """
    raw_results: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        if end - start < MIN_SEGMENT_SEC:
            raw_results.append({"seg_idx": i, "speaker": speaker, "prosody": None})
            continue

        prosody = analyze_prosody_for_segment(audio_path, start, end)
        if not _is_valid_prosody(prosody):
            raw_results.append({"seg_idx": i, "speaker": speaker, "prosody": None})
        else:
            raw_results.append({"seg_idx": i, "speaker": speaker, "prosody": prosody})

    baselines = _compute_speaker_baselines(raw_results)
    recent_by_speaker: Dict[str, List[Dict[str, Any]]] = {}

    segment_results: List[Dict[str, Any]] = []
    for item in raw_results:
        speaker = item["speaker"]
        prosody = item.get("prosody")
        if not _is_valid_prosody(prosody) or speaker not in baselines:
            segment_results.append(item)
            continue

        recent = recent_by_speaker.get(speaker, [])[-ROLLING_WINDOW:]
        enriched = _enrich_prosody(prosody, baselines[speaker], recent)
        segment_results.append({
            "seg_idx": item["seg_idx"],
            "speaker": speaker,
            "prosody": enriched,
        })
        recent_by_speaker.setdefault(speaker, []).append(enriched)

    speaker_prosody = _summarize_by_speaker(segment_results, baselines)
    logger.info(
        f"프로소디 분석 완료: {len(segment_results)}개 세그먼트, "
        f"{len(speaker_prosody)}명 화자 (baseline+급변 적용)"
    )
    return segment_results, speaker_prosody


def _summarize_by_speaker(
    segment_results: List[Dict[str, Any]],
    baselines: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    by_speaker: Dict[str, List[Dict]] = {}
    for item in segment_results:
        if not _is_valid_prosody(item.get("prosody")):
            continue
        by_speaker.setdefault(item["speaker"], []).append(item["prosody"])

    result = {}
    for speaker, prosodies in by_speaker.items():
        baseline = baselines.get(speaker, {})
        sudden_count = sum(1 for p in prosodies if p.get("sudden_change"))

        avg_pitch = float(np.mean([p["pitch_mean"] for p in prosodies]))
        pitch_var = float(np.mean([p["pitch_std"] for p in prosodies]))
        avg_energy = float(np.mean([p["energy_mean"] for p in prosodies]))

        rel_counts = Counter([p.get("relative_pitch", "normal") for p in prosodies])
        var_counts = Counter([p.get("pitch_var_level", "normal") for p in prosodies])
        energy_levels = Counter([p["energy_level"] for p in prosodies])
        rate_levels = Counter([p["speech_rate"] for p in prosodies])

        dominant_rel = rel_counts.most_common(1)[0][0]
        dominant_var = var_counts.most_common(1)[0][0]
        energy_lv = energy_levels.most_common(1)[0][0]
        rate_lv = rate_levels.most_common(1)[0][0]

        rel_ko = _REL_PITCH_KO.get(dominant_rel, "평소")
        var_ko = _VAR_KO.get(dominant_var, "보통")
        energy_desc = {"low": "기력 없는 톤", "normal": "안정적인 톤", "high": "활기찬 톤"}
        rate_desc = {"slow": "느린 말속도", "normal": "보통 속도", "fast": "빠른 말속도"}

        description = (
            f"평소피치 {baseline.get('baseline_pitch', 0):.0f}Hz, "
            f"전체 {rel_ko}, 변동 {var_ko}, "
            f"{energy_desc.get(energy_lv, '')}, {rate_desc.get(rate_lv, '')}"
        )
        if sudden_count > 0:
            description += f", 급변 {sudden_count}회"

        result[speaker] = {
            "baseline_pitch": round(baseline.get("baseline_pitch", 0), 1),
            "baseline_energy": round(baseline.get("baseline_energy", 0), 4),
            "avg_pitch": round(avg_pitch, 1),
            "pitch_variability": round(pitch_var, 1),
            "avg_energy": round(avg_energy, 4),
            "dominant_relative_pitch": dominant_rel,
            "dominant_pitch_var_level": dominant_var,
            "energy_level": energy_lv,
            "speech_rate": rate_lv,
            "sudden_change_count": sudden_count,
            "voice_description": description,
        }

    return result


def analyze_prosody_by_speaker(
    audio_path: str,
    segments: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """화자별 프로소디 종합 (analyze_prosody_full 래퍼)"""
    _, speaker_prosody = analyze_prosody_full(audio_path, segments)
    return speaker_prosody


def analyze_prosody_per_segment(
    audio_path: str,
    segments: List[Dict]
) -> List[Dict[str, Any]]:
    """세그먼트별 운율 (analyze_prosody_full 래퍼)"""
    segment_prosody, _ = analyze_prosody_full(audio_path, segments)
    return segment_prosody
