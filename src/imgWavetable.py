import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal as signal

import ctcsound
import sys
from pathlib import Path
import time
import os

import random
random.seed(time.time())

import config

# ---------------------------------------------------------------------------
# Step 1: Load Image with opencv

IMG_DIR = "img"

# Images should be placed in src/img/
image_path = os.path.join(IMG_DIR, config.IMAGE_PATH)
imageColor = cv.imread(image_path, cv.IMREAD_COLOR)
if imageColor is None:
    print(f"Couldn't load color image at {image_path}!", file=sys.stderr)
    sys.exit(1)

imageGrayScale = cv.cvtColor(imageColor, cv.COLOR_BGR2GRAY)

if imageGrayScale is None:
    print(f"Couldn't load image!", file=sys.stderr)
    sys.exit(1)

height, width = imageGrayScale.shape[:2]

# Global variables for visualization
activeRows = {}  # {row_num: (start_time, note_length, voice_id, pan, note_length)}

def selectRandomRow():
    # Select a row
    rowNum = random.randrange(0, height, 1)
    print(f"Playing row: {rowNum}")
    row = imageGrayScale[rowNum, :]
    return row, rowNum

def fftRow(row):
    # ---------------------------------------------------------------------------
    # Step 2: Compute DFT and Create Waveform
    # Compute FFT
    dft_row = np.fft.fft(row)
    # Remove high frequencies
    if (config.FFT_CUTOFF_RATIO > 0):
        dft_row[(int)(width/config.FFT_CUTOFF_RATIO):] = 0
    # Convert back to time domain
    waveform = np.real(np.fft.ifft(dft_row))

    # Normalize to 1024 samples
    targetLen = config.TARGET_WAVEFORM_LENGTH
    if len(waveform) != targetLen:
        waveform = signal.resample(waveform, targetLen)

    # Normalize waveform to [-1, 1] range
    waveform = waveform - np.min(waveform)
    maxValue = np.max(np.abs(waveform))
    if(maxValue != 0):
        waveform = waveform / maxValue
    waveform = (waveform - 0.5) * 2

    # Apply hanning window to waveform for 0 crossings
    hanningWindow = np.hanning(len(waveform))
    windowedWaveform = waveform * hanningWindow
    return windowedWaveform

def visualize(row, windowedWaveform):
    # ---------------------------------------------------------------------------
    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Image Row (Pixel Values)")
    plt.plot(row)
    plt.subplot(1, 2, 2)
    plt.title("Generated Waveform")
    plt.plot(windowedWaveform)  # Show first 100 samples
    plt.tight_layout()
    plt.show()

def drawImage():
    # Image Visualization
    # Create a copy of the color image
    display_image = imageColor.copy()
    
    current_time = time.time()
    rows_to_remove = []
    
    # Draw yellow highlights for active rows
    for row_num, (start_time, note_length, voice_id) in activeRows.items():
        # Check if note is still playing
        if current_time - start_time < note_length:
            # Draw yellow line across the row
            cv.line(display_image, (0, row_num), (width-1, row_num), (0, 0, 255), 2)
            
            # Add voice ID text
            cv.putText(display_image, f"V{voice_id}", (10, row_num - 5), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Note is finished, mark for removal
            rows_to_remove.append(row_num)
    
    # Remove finished notes
    for row_num in rows_to_remove:
        del activeRows[row_num]
    
    # Display the image
    cv.imshow('imgWavetable - Active Rows', display_image)

# ------------------------------------------------------------------------
# Csound init
cs = ctcsound.Csound()
csd = Path("cs.csd")

# Convert to absolute path (recommended for reliability) and then to string
result = cs.compile_csd(str(csd.absolute()), 0)

if result != ctcsound.CSOUND_SUCCESS:
    print(f"Error compiling csd!", file=sys.stderr)
    sys.exit(1)

# Start performance thread
csThread = ctcsound.CsoundPerformanceThread(cs.csound())

# Start the engine    
result = cs.start()

if result != ctcsound.CSOUND_SUCCESS:
    print(f"Error starting Csound!", file=sys.stderr)
    sys.exit(1)

# Start thread
csThread.play()

# ------------------------------------------------------------------------
# API Interaction

class voice:
    def __init__(self, tableNum, baseNoteLength, amp, pan, lengthDev, overlap = 0.8):
        self.tableNum = tableNum
        self.baseNoteLength = baseNoteLength
        self.lastTime = time.time()
        self.overlap = overlap
        self.amp = amp
        self.pan = pan
        self.lengthDev = lengthDev
        self.newNoteLength()

    def newNoteLength(self) -> None:
        self.nextNote = self.baseNoteLength * random.uniform(1-self.lengthDev, 1+self.lengthDev)
        # ensure next note length isn't less than 0
        self.nextNote = max(self.nextNote, 0.01)
        self.noteLength = self.nextNote * self.overlap

    def checkTimeAndPlay(self):
        currentTime = time.time()
        if currentTime - self.lastTime >= self.nextNote:
            self.playRandomRow()
            self.lastTime = currentTime

    def playRandomRow(self):
        row, rowNum= selectRandomRow()
        waveform = fftRow(row)
        self.newNoteLength()
        self.fillTable(waveform)
        self.playNote()

        activeRows[rowNum] = (time.time(), self.noteLength, self.tableNum)

    def fillTable(self, waveform):
        # Get pointer to table
        table = cs.table(self.tableNum)
        table *= 0 # clear table
        table += waveform

    def playNote(self):
        # Get random pitch in freq range
        mult = random.uniform(config.FREQUENCY_RANGE[0], config.FREQUENCY_RANGE[1])
        startPitch = config.BASE_FREQUENCY * mult
        
        # Get random offset of start freq
        octave_offset = random.uniform(config.PITCH_GLIDE_RANGE[0], config.PITCH_GLIDE_RANGE[1])
        endPitch = startPitch * (2 ** octave_offset)

        # Send score statement to CSound
        print(f"Start Pitch: {startPitch}, End Pitch: {endPitch}, Pan: {self.pan}")
        cs.event_string(f"i 1 0 {self.noteLength} {self.tableNum} {startPitch} {endPitch} {self.amp} {self.pan}")


def createVoices(count, minNoteLen, maxNoteLen, lengthDev, overlap):
    """Creates and returns an array of instances of the voice class"""
    cs.event_string(f"i99 0 0 {count}")
    instances = []
    #noteLenStep = (maxNoteLen - minNoteLen) / (count - 1)
    if(config.AMP == -1):
        amp = 1 / count
    else:
        amp = config.AMP

    noteLenValues = [i * (maxNoteLen - minNoteLen) / (count - 1) for i in range(count)] if count > 1 else [(maxNoteLen-minNoteLen)/2]
    random.shuffle(noteLenValues)

    # Create equidistant pan values (0 to 1)
    panValues = [i / (count - 1) for i in range(count)] if count > 1 else [0.5]
    # Shuffle the pan values randomly
    random.shuffle(panValues)

    for i in range(count):
        instances.append(voice(i + 1, minNoteLen + noteLenValues[i], amp, panValues[i], lengthDev, overlap))
    return instances

voices = createVoices(config.VOICE_COUNT, config.MIN_NOTE_LENGTH, config.MAX_NOTE_LENGTH, config.NOTE_LENGTH_DEVIATION, config.VOICE_OVERLAP)

for voice in voices:
    voice.checkTimeAndPlay()

while 1:
    # Audio Processing
    for voice in voices:
        voice.checkTimeAndPlay()

    drawImage()

    # break out if escape key is pressed
    k = cv.waitKey(50) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
csThread.stop()      # Tell Csound to stop
csThread.join()      # Wait for it to finish
sys.exit()