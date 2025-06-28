"""
Configuration file for imgWavetable synthesis parameters
"""

# Image processing settings
IMAGE_PATH = 'WhiteDiag.jpg'
FFT_CUTOFF_RATIO = 100  # Remove frequencies above width/FFT_CUTOFF_RATIO
TARGET_WAVEFORM_LENGTH = 1024

# Voice settings
VOICE_COUNT = 30
MIN_NOTE_LENGTH = 3.0 # each voice will be assigned a note length equally distributed in the note length range
MAX_NOTE_LENGTH = 8.0
VOICE_OVERLAP = 1
NOTE_LENGTH_DEVIATION = 0.5 # 0-1, percentage the note length can deviate from the voice's note length
AMP = -1 # 0-1, overrides automatic gain adjustment based on voice. Set to -1 so scale amp by 1/VOICE_COUNT

# Audio settings
BASE_FREQUENCY = 400.0
FREQUENCY_RANGE = (0.25, 5)  # Multiplier range for base frequency
PITCH_GLIDE_RANGE = (-4, 4)  # Octave range for pitch gliding