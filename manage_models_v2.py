from torch import nn
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import numpy as np
import librosa

# Wav2Vec2 Model definition
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_labels):
        super(EmotionRecognitionModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

# CNN Model with Global Average Pooling
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.85)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Custom Audio Augmentations
def add_gaussian_noise(audio, amplitude=0.015):
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, amplitude, len(audio))
        return audio + noise
    return audio

def pitch_shift(audio, sr, n_steps):
    if np.random.rand() < 0.5:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return audio

def time_stretch(audio, rate):
    if np.random.rand() < 0.5:
        return librosa.effects.time_stretch(audio, rate=rate)
    return audio

def apply_augmentations(audio, sr):
    audio = add_gaussian_noise(audio, amplitude=np.random.uniform(0.005, 0.015))
    audio = pitch_shift(audio, sr, n_steps=np.random.uniform(-4, 4))
    audio = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    return audio

# Feature Extraction with Normalization
def extract_features(file_path, augment=True):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) < 0.5 * sr:  # Skip clips <0.5s
            print(f"Skipping {file_path}: Too short")
            return None
        if augment:
            y = apply_augmentations(y, sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel = mel_spectrogram_db.T
        if mel.shape[0] > 128:
            mel = mel[:128, :]
        elif mel.shape[0] < 128:
            mel = np.pad(mel, ((0, 128 - mel.shape[0]), (0, 0)), mode='constant')
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def ml_model_extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0) #MFCC (Mel-frequency cepstral coefficients) – captures timbral texture.
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0) #Chroma – reflects pitch class energy (musical notes).
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0) #Spectral Contrast – helps with timbre and brightness.
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0) #ZCR (Zero Crossing Rate) – good for detecting noisiness or fricatives.
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0) #RMS (Root Mean Square Energy) – correlates with loudness.

    return np.hstack([mfcc, chroma, contrast, zcr, rms])


import librosa
import soundfile as sf
import numpy as np
import os

def convert_wav_audio(file_path, target_sr=16000, output_dir="test/normalised"):
    """
    Load a WAV file, fix format issues (e.g., convert stereo to mono), and save the corrected file in the output directory.
    Ensures the output is a 1D (mono) or 2D (batched mono) array with shape [length] or [1, length].
    
    Args:
        file_path (str): Path to the input WAV file.
        target_sr (int): Target sample rate (default: 16000 Hz).
        output_dir (str): Directory to save the corrected WAV file (default: 'test/normalised').
    
    Returns:
        tuple: (audio_data, sample_rate, output_path)
            - audio_data: NumPy array of shape [length] (mono) or [1, length] (batched mono).
            - sample_rate: Sample rate of the audio.
            - output_path: Path to the saved corrected WAV file.
    
    Raises:
        ValueError: If the file is not a valid WAV or cannot be processed.
    """
    try:
        # Validate input file existence
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file does not exist: {file_path}")
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        
        # Check if audio is stereo (2 channels)
        needs_conversion = audio.ndim > 1 and audio.shape[0] == 2
        
        if needs_conversion:
            # Convert stereo to mono by averaging channels
            audio = np.mean(audio, axis=0)
        
        # Ensure audio is 1D (mono)
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Reshape to [1, length] for batched conv1d compatibility
        audio = audio.reshape(1, -1) if audio.ndim == 1 else audio
        
        # Validate shape for conv1d (should be [1, length] or [length])
        if audio.ndim not in [1, 2] or (audio.ndim == 2 and audio.shape[0] != 1):
            raise ValueError(f"Unexpected audio shape after processing: {audio.shape}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path with 'corrected_' prefix to avoid overwriting
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"{file_name}")
        
        # Save the corrected audio (use 1D array for writing)
        sf.write(output_path, audio.squeeze(), sr)
        
        print(f"Corrected audio saved to: {output_path}")
        return audio, sr, output_path
    
    except Exception as e:
        # Fallback to soundfile if librosa fails
        try:
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            needs_conversion = audio.ndim > 1 and audio.shape[1] == 2
            if needs_conversion:
                audio = np.mean(audio, axis=1)
            
            # Ensure 1D or [1, length]
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            # Resample if necessary
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Reshape to [1, length]
            audio = audio.reshape(1, -1) if audio.ndim == 1 else audio
            
            if audio.ndim not in [1, 2] or (audio.ndim == 2 and audio.shape[0] != 1):
                raise ValueError(f"Unexpected audio shape after processing: {audio.shape}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output file path with 'corrected_' prefix
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{file_name}")
            
            # Save the corrected audio
            sf.write(output_path, audio.squeeze(), sr)
            
            print(f"Corrected audio saved to: {output_path}")
            return audio, sr, output_path
        
        except Exception as e2:
            raise ValueError(f"Failed to process WAV file: {str(e)} | {str(e2)}")
