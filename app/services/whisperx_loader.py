"""
STT + 화자분리 + 단어 타임스탬프 (GB10 Blackwell 호환)
- STT: HuggingFace transformers Whisper (PyTorch 네이티브)
- 정렬: whisperx align (PyTorch)
- 화자분리: pyannote via whisperx (PyTorch)
"""
import os
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Dict, List, Any
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_AUTH_TOKEN", "")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


YOUTUBE_BLACKLIST = [
    "한글자막 제공",
    "자막 제공 및 협조",
    "다음 영상에서 만나요",
    "구독과 좋아요",
    "진심으로 감사드립니다",
    "좋아요와 구독",
    "채널을 구독",
    "영상이 도움이 되셨다면",
]


def _is_hallucination(text: str) -> bool:
    """Whisper 반복 환각 감지"""
    if not text or len(text) < 3:
        return True

    words = text.split()
    if len(words) > 5 and len(set(words)) <= 2:
        return True

    if len(words) > 10:
        from collections import Counter
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        most_common_count = Counter(bigrams).most_common(1)[0][1]
        if most_common_count > len(bigrams) * 0.6:
            return True

    # 문자 단위 반복 감지 (공백 없는 반복: "공지부공지부공지부...")
    stripped = text.replace(" ", "")
    if len(stripped) > 12:
        for n in range(2, 7):
            from collections import Counter
            ngrams = [stripped[i:i+n] for i in range(len(stripped) - n + 1)]
            top_count = Counter(ngrams).most_common(1)[0][1]
            coverage = (top_count * n) / len(stripped)
            if coverage > 0.5:
                return True

    return False


def _clean_youtube_hallucination(text: str) -> str:
    """YouTube 학습 데이터 환각 문구를 제거하고 정제된 텍스트 반환"""
    for phrase in YOUTUBE_BLACKLIST:
        if phrase in text:
            text = text.replace(phrase, "")
    text = text.strip()
    text = " ".join(text.split())
    return text


def _validate_text_duration(text: str, duration_sec: float) -> str:
    """타임스탬프 대비 텍스트량 검증. 초과분 잘라냄."""
    if duration_sec <= 0:
        return text
    chars_per_sec = len(text.replace(" ", "")) / duration_sec
    if chars_per_sec > 15:
        max_chars = int(duration_sec * 10)
        if max_chars < len(text):
            cutoff = text[:max_chars].rfind(" ")
            if cutoff > max_chars * 0.5:
                text = text[:cutoff]
            else:
                text = text[:max_chars]
            logger.info(f"[필터] 텍스트량 초과 잘라냄 ({chars_per_sec:.0f}자/초 → {len(text)}자)")
    return text


def _text_similarity(a: str, b: str) -> float:
    """두 텍스트의 단어 기반 유사도 (0~1). Jaccard 유사도 사용."""
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def transcribe_with_diarization(
    audio_path: str,
    min_speakers: int = 2,
    max_speakers: int = 4
) -> Dict[str, Any]:
    """
    전체 파이프라인: STT → 정렬 → 화자분리

    Returns:
        {
            "segments": [{
                "start": 0.5, "end": 3.2,
                "text": "어르신 안녕하세요",
                "speaker": "SPEAKER_00"
            }, ...],
            "transcript_plain": "전체 텍스트",
            "transcript_labeled": "[00:00.5 → 00:03.2] [SPEAKER_00] 어르신 안녕하세요\n..."
        }
    """
    device = get_device()
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. STT (transformers Whisper)
    logger.info(f"[STT] 시작: {Path(audio_path).name} (device={device})")
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=(4, 2),
        return_timestamps=True,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"language": "korean", "task": "transcribe"},
        return_timestamps=True,
        batch_size=8,
    )

    chunks = result.get("chunks", [])
    logger.info(f"[STT] 완료: {len(chunks)}개 청크")

    segments = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        text = chunk.get("text", "").strip()

        if _is_hallucination(text):
            logger.info(f"[STT] 환각 필터링: '{text[:30]}...'")
            continue

        text = _clean_youtube_hallucination(text)
        if not text or len(text) < 3:
            logger.info(f"[STT] YouTube 환각 제거 후 빈 세그먼트 스킵")
            continue

        start = ts[0] if ts[0] is not None else 0.0
        end = ts[1] if ts[1] is not None else 0.0
        if end <= start:
            continue

        duration = end - start
        text = _validate_text_duration(text, duration)
        if not text or len(text) < 3:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": text
        })

    del model, pipe
    torch.cuda.empty_cache()

    # 2. 단어 정렬 (whisperx align)
    logger.info("[정렬] 단어 타임스탬프 정렬 시작")
    audio_arr = whisperx.load_audio(audio_path)
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ko", device=str(device)
    )
    aligned = whisperx.align(
        segments, align_model, align_metadata,
        audio_arr, device=str(device), return_char_alignments=False
    )
    aligned_segments = aligned.get("segments", segments)
    logger.info("[정렬] 완료")

    del align_model
    torch.cuda.empty_cache()

    # 3. 화자분리 (pyannote)
    if not HF_TOKEN:
        logger.warning("HF_AUTH_TOKEN 없음 - 화자분리 스킵")
        final_segments = aligned_segments
    else:
        logger.info("[화자분리] 시작")
        diarize_pipeline = DiarizationPipeline(
            token=HF_TOKEN, device=str(device)
        )
        diarize_result = diarize_pipeline(
            audio_arr, min_speakers=min_speakers, max_speakers=max_speakers
        )
        final = whisperx.assign_word_speakers(
            diarize_result, {"segments": aligned_segments}
        )
        final_segments = final.get("segments", aligned_segments)
        logger.info("[화자분리] 완료")

        del diarize_pipeline
        torch.cuda.empty_cache()

    # 4. 같은 화자 인접 세그먼트 병합 (갭 1.5초 이내 or 세그먼트 2초 미만)
    merged_segments = []
    for seg in final_segments:
        if not merged_segments:
            merged_segments.append(dict(seg))
            continue

        prev = merged_segments[-1]
        same_speaker = prev.get("speaker") == seg.get("speaker")
        gap = seg.get("start", 0) - prev.get("end", 0)
        prev_duration = prev.get("end", 0) - prev.get("start", 0)
        cur_duration = seg.get("end", 0) - seg.get("start", 0)

        if same_speaker and (gap < 1.5 or prev_duration < 2.0 or cur_duration < 2.0):
            prev["end"] = seg.get("end", prev["end"])
            prev_text = prev.get("text", "").strip()
            cur_text = seg.get("text", "").strip()
            if cur_text and cur_text != prev_text:
                prev["text"] = prev_text + " " + cur_text
        else:
            merged_segments.append(dict(seg))

    if len(merged_segments) < len(final_segments):
        logger.info(f"[병합] 인접 세그먼트 병합: {len(final_segments)} → {len(merged_segments)}")
    final_segments = merged_segments

    # 5. 세그먼트 간 반복 필터 (동일/유사 문장 3회 이상 연속 → 첫 2개만 유지)
    deduped_segments = []
    repeat_count = 1
    for i, seg in enumerate(final_segments):
        text = seg.get("text", "").strip()
        prev_text = final_segments[i - 1].get("text", "").strip() if i > 0 else ""

        if text == prev_text or _text_similarity(text, prev_text) > 0.8:
            repeat_count += 1
            if repeat_count >= 3:
                logger.info(f"[필터] 세그먼트 간 반복 제거 ({repeat_count}회): '{text[:30]}...'")
                continue
        else:
            repeat_count = 1
        deduped_segments.append(seg)

    if len(deduped_segments) < len(final_segments):
        removed = len(final_segments) - len(deduped_segments)
        logger.info(f"[필터] 세그먼트 간 반복 {removed}개 제거 ({len(final_segments)} → {len(deduped_segments)})")
    final_segments = deduped_segments

    # 6. 출력 포맷팅
    labeled_lines = []
    plain_lines = []

    for seg in final_segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()

        start_str = f"{int(start//60):02d}:{start%60:06.3f}"
        end_str = f"{int(end//60):02d}:{end%60:06.3f}"

        labeled_lines.append(f"[{start_str} --> {end_str}] [{speaker}] {text}")
        plain_lines.append(f"[{start_str} --> {end_str}] {text}")

    return {
        "segments": final_segments,
        "transcript_plain": "\n".join(plain_lines),
        "transcript_labeled": "\n".join(labeled_lines)
    }
