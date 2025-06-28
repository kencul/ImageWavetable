# imgWavetable

A creative audio synthesis project that converts image data into musical wavetables using FFT processing and Csound.

## Overview

imgWavetable transforms visual information from images into unique audio textures by:
1. Extracting random rows from grayscale images
2. Processing the pixel data through FFT (Fast Fourier Transform)
3. Converting the frequency-domain data back to time-domain waveforms
4. Using these waveforms as wavetables in a polyphonic Csound synthesis engine

## Features

- **Image-to-Audio Conversion**: Converts image rows into musical waveforms
- **Multi-Voice Polyphony**: 30 independent voices with different timing and panning
- **Real-time Synthesis**: Live audio generation with Csound
- **Rich Audio Processing**: Includes reverb, delay, FM synthesis, and stereo panning
- **Graceful Shutdown**: Proper cleanup and signal handling
- **Configurable Parameters**: Easy-to-modify settings in `config.py`

## Requirements

- Python 3.8+
- Csound 7
- Audio output device

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd imgWavetable
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Csound is installed on your system:
   - **macOS**: `brew install csound`
   - **Ubuntu/Debian**: `sudo apt-get install csound`
   - **Windows**: Download from [csound.com](https://csound.com/download.html)

## Usage

1. **Place your images in the `src/img/` directory.**
   - Example: `src/img/rock.jpg`, `src/img/starrynight.jpg`
2. **Set the image filename in `src/config.py`:**
   ```python
   IMAGE_PATH = 'rock.jpg'  # Just the filename, not the path
   ```
3. **Run the synthesis:**
   ```bash
   cd src
   python imgWavetable.py
   ```
4. The system will start generating audio based on the image data and open a window showing the image with highlighted rows.
5. Press `Esc` to stop the synthesis and close the window.

## Configuration

Edit `src/config.py` to customize the synthesis parameters:

- **IMAGE_PATH**: Filename of the image to use (must be in `src/img/`)
- **Voice Count**: Number of simultaneous voices (default: 30)
- **Note Lengths**: Range of note durations (default: 8-15 seconds)
- **Frequency Range**: Pitch variation range
- **FFT Cutoff**: High-frequency filtering ratio

## How It Works

### Image Processing
1. Loads a color image from `src/img/` using OpenCV
2. Converts it to grayscale for audio processing
3. Randomly selects rows from the image
4. Converts pixel values to audio samples

### FFT Processing
1. Applies FFT to convert pixel data to frequency domain
2. Removes high frequencies to create smoother waveforms
3. Converts back to time domain using inverse FFT
4. Normalizes and applies Hanning window for smooth transitions

### Audio Synthesis
1. Creates multiple voice instances with different parameters
2. Each voice generates waveforms from different image rows
3. Uses Csound for real-time audio synthesis with:
   - FM synthesis with pitch gliding
   - Stereo panning
   - Reverb and delay effects
   - Dynamic amplitude envelopes

### Visualization
- A window displays the image with highlighted rows in red while they are being played.
- Each active row is labeled with its voice number (e.g., V1, V2, ...).
- Highlights appear when a voice starts playing a note and disappear when the note finishes.

## File Structure

```
imgWavetable/
├── src/
│   ├── imgWavetable.py    # Main synthesis engine
│   ├── config.py          # Configuration parameters
│   ├── cs.csd             # Csound synthesis definition
│   ├── img/               # Directory for image files
│   │   └── yourimagefile.jpg
├── requirements.txt       # Python dependencies
└── README.md              # This file
```