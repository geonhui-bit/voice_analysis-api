"""
음성 분석 단독 테스트 스크립트 (GB10 Blackwell 호환)
- STT: transformers Whisper (PyTorch 네이티브, CTranslate2 없이)
- 화자분리: pyannote.audio (PyTorch)
- 감정분석: SpeechBrain (PyTorch)
- 피치분석: torchcrepe (PyTorch) + librosa
- 전부 CUDA SM 12.1 에서 동작

사용법: python test_audio_analysis.py /path/to/audio.wav
"""
import sys
import json
import time
import os
import torch
import numpy as np
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def test_whisper_transformers(audio_path: str):
    """1단계: HuggingFace transformers Whisper STT + pyannote 화자분리"""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = get_device()
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"\n{'='*60}")
    print(f"[1/4] Whisper STT (transformers) + 화자분리")
    print(f"  device: {device}, dtype: {torch_dtype}")
    print(f"  audio: {audio_path}")
    print(f"{'='*60}")

    # --- STT (transformers pipeline) ---
    t0 = time.time()
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
        return_timestamps=True,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"language": "korean", "task": "transcribe"},
        return_timestamps=True,
    )

    t_stt = time.time() - t0
    chunks = result.get("chunks", [])
    print(f"  STT 완료: {len(chunks)}개 청크 ({t_stt:.1f}초)")

    # transformers 결과를 whisperx 정렬 포맷으로 변환
    segments = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        segments.append({
            "start": ts[0] if ts[0] is not None else 0.0,
            "end": ts[1] if ts[1] is not None else 0.0,
            "text": chunk.get("text", "").strip()
        })

    # STT 모델 메모리 해제
    del model, pipe
    torch.cuda.empty_cache()

    # --- 단어 정렬 (whisperx align, PyTorch 기반) ---
    t0 = time.time()
    audio_arr = whisperx.load_audio(audio_path)
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ko", device=str(device)
    )
    aligned = whisperx.align(
        segments, align_model, align_metadata,
        audio_arr, device=str(device), return_char_alignments=False
    )
    aligned_segments = aligned.get("segments", segments)
    t_align = time.time() - t0
    print(f"  단어 정렬 완료 ({t_align:.1f}초)")

    del align_model
    torch.cuda.empty_cache()

    # --- 화자분리 (pyannote, PyTorch 기반) ---
    hf_token = os.getenv("HF_AUTH_TOKEN", "")
    if not hf_token:
        print("  ⚠ HF_AUTH_TOKEN 없음 - 화자분리 스킵")
        return aligned_segments, audio_path

    t0 = time.time()
    diarize_pipeline = DiarizationPipeline(
        token=hf_token, device=str(device)
    )
    diarize_result = diarize_pipeline(audio_arr, min_speakers=2, max_speakers=4)
    final = whisperx.assign_word_speakers(diarize_result, {"segments": aligned_segments})
    final_segments = final.get("segments", aligned_segments)
    t_diar = time.time() - t0
    print(f"  화자분리 완료 ({t_diar:.1f}초)")

    del diarize_pipeline
    torch.cuda.empty_cache()

    # 결과 미리보기
    print(f"\n  --- 처음 10개 세그먼트 ---")
    for seg in final_segments[:10]:
        speaker = seg.get("speaker", "?")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        print(f"  [{start:07.3f} -> {end:07.3f}] [{speaker}] {text}")
    if len(final_segments) > 10:
        print(f"  ... 외 {len(final_segments)-10}개 더")

    return final_segments, audio_path


def test_emotion(audio_path: str, segments: list):
    """2단계: SpeechBrain 감정분석 (PyTorch 기반)"""
    print(f"\n{'='*60}")
    print(f"[2/4] SpeechBrain 감정분석")
    print(f"{'='*60}")

    try:
        import torchaudio
        from speechbrain.pretrained.interfaces import foreign_class

        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir="pretrained_models/emotion"
        )

        from collections import defaultdict
        speaker_segs = defaultdict(list)
        for seg in segments:
            if seg.get("end", 0) - seg.get("start", 0) > 2.0:
                speaker_segs[seg.get("speaker", "?")].append(seg)

        results = {}
        for speaker, segs in speaker_segs.items():
            sample_segs = segs[:3]
            emotions = []

            for seg in sample_segs:
                start, end = seg["start"], seg["end"]
                waveform, sr = torchaudio.load(audio_path)
                chunk = waveform[:, int(start*sr):int(end*sr)]

                if sr != 16000:
                    chunk = torchaudio.transforms.Resample(sr, 16000)(chunk)

                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    torchaudio.save(f.name, chunk, 16000)
                    out_prob, score, index, text_lab = classifier.classify_file(f.name)
                    emotions.append(text_lab[0] if text_lab else "?")
                    Path(f.name).unlink()

            results[speaker] = emotions
            print(f"  {speaker}: {emotions}")

        return results

    except Exception as e:
        print(f"  ❌ 감정분석 실패: {e}")
        import traceback; traceback.print_exc()
        return {}


def test_prosody(audio_path: str, segments: list):
    """3단계: torchcrepe 피치 + librosa 에너지 분석 (PyTorch 기반)"""
    print(f"\n{'='*60}")
    print(f"[3/4] torchcrepe 피치 + librosa 에너지 분석")
    print(f"{'='*60}")

    try:
        import librosa
        import torchcrepe
        from collections import defaultdict

        device = get_device()

        speaker_segs = defaultdict(list)
        for seg in segments:
            if seg.get("end", 0) - seg.get("start", 0) > 2.0:
                speaker_segs[seg.get("speaker", "?")].append(seg)

        results = {}
        for speaker, segs in speaker_segs.items():
            sample_segs = segs[:3]
            pitches = []
            energies = []

            for seg in sample_segs:
                start, end = seg["start"], seg["end"]
                y, sr = librosa.load(
                    audio_path, sr=16000, offset=start, duration=end-start
                )

                # 피치 (torchcrepe, PyTorch GPU 사용)
                audio_tensor = torch.tensor(y).unsqueeze(0).to(device)
                frequency = torchcrepe.predict(
                    audio_tensor, 16000,
                    hop_length=160,
                    fmin=50, fmax=500,
                    model='tiny',
                    decoder=torchcrepe.decode.viterbi,
                    device=device,
                    return_periodicity=False,
                    batch_size=1024
                )
                freq_np = frequency.cpu().numpy().flatten()
                valid = freq_np[(freq_np > 50) & (freq_np < 500)]
                if len(valid) > 0:
                    pitches.append(float(np.mean(valid)))

                # 에너지 (librosa RMS, CPU)
                rms = librosa.feature.rms(y=y)[0]
                energies.append(float(np.mean(rms)))

            avg_pitch = np.mean(pitches) if pitches else 0
            avg_energy = np.mean(energies) if energies else 0

            results[speaker] = {
                "avg_pitch_hz": round(float(avg_pitch), 1),
                "avg_energy": round(float(avg_energy), 4)
            }
            print(f"  {speaker}: pitch={avg_pitch:.1f}Hz, energy={avg_energy:.4f}")

        return results

    except Exception as e:
        print(f"  ❌ 프로소디 분석 실패: {e}")
        import traceback; traceback.print_exc()
        return {}


def test_guard_filter(segments: list):
    """4단계: 가드레일 (역할 매핑) 테스트"""
    print(f"\n{'='*60}")
    print(f"[4/4] 가드레일: 화자 역할 자동 매핑")
    print(f"{'='*60}")

    MANAGER_KEYWORDS = [
        "어르신", "혈압", "재볼까", "드셨어요", "하셨어요",
        "다음에", "가져올게", "제가", "드릴게", "확인",
        "운동", "약", "식사", "가르쳐"
    ]

    speaker_scores = {}
    for seg in segments:
        speaker = seg.get("speaker", "?")
        text = seg.get("text", "")

        if speaker not in speaker_scores:
            speaker_scores[speaker] = 0

        for kw in MANAGER_KEYWORDS:
            if kw in text:
                speaker_scores[speaker] += 1

        if text.strip().endswith(("요?", "세요?", "까요?", "나요?", "죠?")):
            speaker_scores[speaker] += 0.5

    sorted_speakers = sorted(
        speaker_scores.items(), key=lambda x: x[1], reverse=True
    )

    roles = {}
    for i, (speaker, score) in enumerate(sorted_speakers):
        if i == 0:
            roles[speaker] = "매니저"
        elif i == 1:
            roles[speaker] = "시니어"
        else:
            roles[speaker] = f"동거인_{i}"
        print(f"  {speaker} -> {roles[speaker]} (score: {score})")

    return roles


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_audio_analysis.py <오디오파일경로>")
        print("예시:   python test_audio_analysis.py ./test_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"❌ 파일 없음: {audio_path}")
        sys.exit(1)

    device = get_device()
    print(f"\n🎙 음성 분석 실험 시작 (GB10 Blackwell 호환)")
    print(f"  파일: {audio_path}")
    print(f"  크기: {Path(audio_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"  SM: {torch.cuda.get_device_capability(0) if device.type == 'cuda' else 'N/A'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if device.type == 'cuda' else "")

    total_start = time.time()

    # 1. Whisper STT + 화자분리
    segments, wav_path = test_whisper_transformers(audio_path)
    if not segments:
        print("\n❌ STT 결과 없음. 중단.")
        sys.exit(1)

    # 2. 감정분석
    emotion_results = test_emotion(wav_path, segments)

    # 3. 프로소디
    prosody_results = test_prosody(wav_path, segments)

    # 4. 가드레일
    roles = test_guard_filter(segments)

    # 최종 결과 저장
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"✅ 전체 완료 ({total_time:.1f}초)")
    print(f"{'='*60}")

    output = {
        "audio_file": audio_path,
        "total_time_sec": round(total_time, 1),
        "num_segments": len(segments),
        "speaker_roles": roles,
        "emotion_results": emotion_results,
        "prosody_results": prosody_results,
        "segments_sample": segments[:20],
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "compute_capability": str(torch.cuda.get_device_capability(0)) if device.type == "cuda" else "N/A",
            "pytorch": torch.__version__
        }
    }

    output_path = f"test_result_{Path(audio_path).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {output_path}")

    print(f"\n  다음 단계: 이 결과가 만족스러우면 LLM 요약 연결")


if __name__ == "__main__":
    main()
