import os
import torch
import librosa
import numpy as np

class AcousticExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._cached_audio = None
        self._cached_file = None

    def _load_model(self):
        if self.model is None:
            from transformers import ClapModel, ClapProcessor
            
            # Check local models/clip directory first (which user confirmed contains the CLAP model)
            local_path = os.path.abspath("./models/clip")
            if not (os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json"))):
                local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/clip"))
                
            if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                model_source = local_path
                print(f"[ACOUSTIC WORKER] Lazy-loading LAION-CLAP (HTSAT-Unfused) model from local directory: {local_path}...")
            else:
                model_source = "laion/clap-htsat-unfused"
                print(f"[ACOUSTIC WORKER] Lazy-loading LAION-CLAP (HTSAT-Unfused) model from Hugging Face Hub: {model_source}...")
                
            self.model = ClapModel.from_pretrained(model_source).to(self.device).eval()
            self.processor = ClapProcessor.from_pretrained(model_source)


    def load_audio_file(self, file_path: str):
        """Loads and caches the 48kHz audio track for the active file to optimize sequential chunk slicing."""
        if self._cached_file != file_path:
            try:
                print(f"[ACOUSTIC WORKER] Loading audio at 48kHz for: {os.path.basename(file_path)}")
                self._cached_audio, _ = librosa.load(file_path, sr=48000)
                self._cached_file = file_path
            except Exception as e:
                print(f"[ACOUSTIC WORKER ERROR] Failed to load audio track: {e}")
                self._cached_audio = None
                self._cached_file = None

    def extract_slice_features(self, start_ts: float, end_ts: float) -> list[float]:
        """Slices the cached audio track and returns the 512-dimensional CLAP embedding."""
        if self._cached_audio is None:
            return [0.0] * 512

        self._load_model()
        try:
            start_idx = int(start_ts * 48000)
            end_idx = int(end_ts * 48000)
            chunk = self._cached_audio[start_idx:end_idx]

            if len(chunk) == 0:
                return [0.0] * 512

            try:
                inputs = self.processor(audio=chunk, sampling_rate=48000, return_tensors="pt").to(self.device)
            except TypeError:
                inputs = self.processor(audios=chunk, sampling_rate=48000, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embed = self.model.get_audio_features(**inputs)
                embed /= embed.norm(dim=-1, keepdim=True)
                return embed.cpu().squeeze(0).tolist()
        except Exception as e:
            print(f"[ACOUSTIC WORKER WARNING] CLAP feature extraction failed: {e}")
            return [0.0] * 512

    def clear_cache(self):
        self._cached_audio = None
        self._cached_file = None
