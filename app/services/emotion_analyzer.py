"""
한국어 음성 감정분석 (wav2vec2-xlsr 기반, 수동 가중치 매핑)
모델: jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance
학습: AI-Hub 한국어 감정 음성 데이터셋, Apache 2.0
감정: 기쁨, 당황, 분노, 불안, 슬픔, 중립 (6가지)

transformers >= 4.51에서 AutoModelForAudioClassification이 이 모델의
커스텀 classifier 가중치를 인식 못하는 문제를 수동 매핑으로 해결.
원본 구조: classifier.dense (1024→1024) → tanh → classifier.out_proj (1024→6)
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from functools import lru_cache
from typing import Dict, List, Any
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

MODEL_ID = "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition2_data_rebalance"

LABELS = ["기쁨", "당황", "분노", "불안", "슬픔", "중립"]

EMOTION_LABELS_KO = {
    "기쁨": "밝음, 긍정적",
    "당황": "당혹감, 혼란",
    "분노": "격앙됨, 불만",
    "불안": "불안, 두려움",
    "슬픔": "우울함, 침울함",
    "중립": "차분함, 담담함",
}


class Wav2Vec2EmotionClassifier(nn.Module):
    """jungjongho 모델의 원본 학습 구조를 재현한 커스텀 클래스"""

    def __init__(self, encoder, dense, out_proj, feature_extractor):
        super().__init__()
        self.encoder = encoder
        self.dense = dense
        self.out_proj = out_proj
        self.feature_extractor = feature_extractor

    @torch.no_grad()
    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        hidden = torch.tanh(self.dense(hidden))
        logits = self.out_proj(hidden)
        return logits


@lru_cache(maxsize=1)
def get_emotion_model():
    """한국어 감정인식 모델 로드 (수동 가중치 매핑)"""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from huggingface_hub import hf_hub_download

    encoder = Wav2Vec2Model.from_pretrained(MODEL_ID)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)

    sd = torch.load(
        hf_hub_download(MODEL_ID, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )

    h = encoder.config.hidden_size  # 1024
    num_labels = len(LABELS)        # 6

    dense = nn.Linear(h, h)
    dense.weight.data.copy_(sd["classifier.dense.weight"])
    dense.bias.data.copy_(sd["classifier.dense.bias"])

    out_proj = nn.Linear(h, num_labels)
    out_proj.weight.data.copy_(sd["classifier.out_proj.weight"])
    out_proj.bias.data.copy_(sd["classifier.out_proj.bias"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2EmotionClassifier(encoder, dense, out_proj, feature_extractor)
    model = model.to(device).eval()

    logger.info(f"한국어 감정인식 모델 로드 완료 (수동 매핑): {MODEL_ID} (device={device})")
    return model, device


def analyze_emotion_for_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float
) -> Dict[str, Any]:
    """
    특정 구간의 감정을 분석합니다.

    Returns:
        {"label": "중립", "score": 0.85, "scores": {"기쁨": 0.05, ...}}
    """
    try:
        waveform, sr = torchaudio.load(audio_path)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_waveform = waveform[:, start_sample:end_sample]

        if segment_waveform.shape[1] < sr * 0.5:
            return {"label": "unknown", "score": 0.0, "scores": {}}

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            segment_waveform = resampler(segment_waveform)

        if segment_waveform.shape[0] > 1:
            segment_waveform = segment_waveform.mean(dim=0, keepdim=True)

        model, device = get_emotion_model()

        inputs = model.feature_extractor(
            segment_waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        logits = model(input_values)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        scores_dict = {}
        for i, prob in enumerate(probs):
            label = LABELS[i] if i < len(LABELS) else f"label_{i}"
            scores_dict[label] = float(prob)

        best_idx = int(np.argmax(probs))
        best_score = float(probs[best_idx])

        if best_score < 0.4:
            best_label = "중립"
        else:
            best_label = LABELS[best_idx] if best_idx < len(LABELS) else "unknown"

        return {
            "label": best_label,
            "score": best_score,
            "scores": scores_dict
        }

    except Exception as e:
        logger.warning(f"감정분석 실패 ({start_sec:.1f}~{end_sec:.1f}s): {e}")
        return {"label": "unknown", "score": 0.0, "scores": {}}


def analyze_emotions_by_speaker(
    audio_path: str,
    segments: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    화자별 감정을 종합 분석합니다.

    Returns:
        {
            "SPEAKER_00": {
                "dominant_emotion": "중립",
                "emotion_distribution": {"기쁨": 0.05, "중립": 0.60, ...},
                "tone_description": "차분함"
            },
            "SPEAKER_01": {...}
        }
    """
    speaker_emotions = {}

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        if end - start < 1.0:
            continue

        emotion = analyze_emotion_for_segment(audio_path, start, end)

        if speaker not in speaker_emotions:
            speaker_emotions[speaker] = []
        speaker_emotions[speaker].append(emotion)

    result = {}
    for speaker, emotions in speaker_emotions.items():
        valid_emotions = [e for e in emotions if e["label"] != "unknown"]

        if not valid_emotions:
            result[speaker] = {
                "dominant_emotion": "unknown",
                "emotion_distribution": {},
                "tone_description": "분석 불가"
            }
            continue

        distribution = {}
        for e in valid_emotions:
            for label, score in e.get("scores", {}).items():
                distribution[label] = distribution.get(label, 0) + score

        total = sum(distribution.values())
        if total > 0:
            distribution = {k: round(v / total, 3) for k, v in distribution.items()}

        sorted_emotions = sorted(distribution.items(), key=lambda x: -x[1])
        dominant = sorted_emotions[0][0]
        dominant_score = sorted_emotions[0][1]

        if len(sorted_emotions) >= 2:
            second = sorted_emotions[1][0]
            second_score = sorted_emotions[1][1]
            gap = dominant_score - second_score
            if gap < 0.10:
                tone = f"{EMOTION_LABELS_KO.get(dominant, dominant)} + {EMOTION_LABELS_KO.get(second, second)} 혼합"
            else:
                tone = EMOTION_LABELS_KO.get(dominant, "차분함")
        else:
            tone = EMOTION_LABELS_KO.get(dominant, "차분함")

        result[speaker] = {
            "dominant_emotion": dominant,
            "emotion_distribution": distribution,
            "tone_description": tone
        }

    logger.info(f"감정분석 완료: {len(result)}명 화자")
    return result
