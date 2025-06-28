import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal as signal

import ctcsound
import sys
from pathlib import Path
import time
import threading

import random
random.seed(time.time())

# ---------------------------------------------------------------------------
# Step 1: Load Image and Extract Row
image = cv.imread('rock.jpg', cv.IMREAD_GRAYSCALE)

if image is None:
    print(f"Couldn't load image!", file=sys.stderr)
    sys.exit(1)

# Load color version for display
image_color = cv.imread('rock.jpg', cv.IMREAD_COLOR)
if image_color is None:
    print(f"Couldn't load color image!", file=sys.stderr)
    sys.exit(1)

height, width = image.shape[:2]

# Global variables for visualization
active_rows = {}  # {row_num: (start_time, note_length, voice_id)}
display_lock = threading.Lock()

def selectRandomRow():
    # Select a row (e.g., middle row)
    rowNum = random.randrange(0, height, 1)
    print(rowNum)
    row = image[rowNum, :]
    return row, rowNum

def fftRow(row):
    # ---------------------------------------------------------------------------
    # Step 2: Compute DFT and Create Waveform
    # Compute FFT
    dft_row = np.fft.fft(row)
    # Remove high frequencies
    dft_row[(int)(width/30):] = 0
    # Convert back to time domain
    waveform = np.real(np.fft.ifft(dft_row))

    # Normalize to 1024 samples
    targetLen = 1024
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

def update_visualization():
    """Update the image visualization with current active rows"""
    global active_rows, image_color
    
    with display_lock:
        # Create a copy of the color image
        display_image = image_color.copy()
        
        current_time = time.time()
        rows_to_remove = []
        
        # Draw yellow highlights for active rows
        for row_num, (start_time, note_length, voice_id) in active_rows.items():
            # Check if note is still playing
            if current_time - start_time < note_length:
                # Draw yellow line across the row
                cv.line(display_image, (0, row_num), (width-1, row_num), (0, 255, 255), 2)
                
                # Add voice ID text
                cv.putText(display_image, f"V{voice_id}", (10, row_num - 5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # Note is finished, mark for removal
                rows_to_remove.append(row_num)
        
        # Remove finished notes
        for row_num in rows_to_remove:
            del active_rows[row_num]
        
        # Display the image
        cv.imshow('imgWavetable - Active Rows', display_image)
        cv.waitKey(1)

def visualization_thread():
    """Thread for updating the visualization"""
    while True:
        try:
            update_visualization()
            time.sleep(0.1)  # Update 10 times per second
        except:
            break

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

# Start visualization thread
vis_thread = threading.Thread(target=visualization_thread, daemon=True)
vis_thread.start()

# ------------------------------------------------------------------------
# API Interaction

class voice:
    def __init__(self, tableNum, noteLength, amp, pan, overlap = 0.8):
        self.tableNum = tableNum
        self.noteLength = noteLength
        self.lastTime = 0 #time.time()
        self.overlap = overlap
        self.amp = amp
        self.pan = pan
    
    def checkTimeAndPlay(self):
        currentTime = time.time()
        if currentTime - self.lastTime >= self.noteLength:
            self.playRandomRow()
            self.lastTime = currentTime

    def fillTable(self, waveform):
        # Get pointer to table
        table = cs.table(self.tableNum)
        table *= 0 # clear table
        table += waveform

    def playNote(self):
        # Get random pitch up or down and octave
        startPitch = 220 * (random.random() * 2 + 0.25)#random.choice([1, 1.3, 1.5, 1.7, 2, 2.3 ,2.5,2.7,3])
        endPitch = startPitch * (pow(2, random.random() * 4 - 2))
        print(f"Start Pitch: {startPitch}, End Pitch: {endPitch}, Pan: {self.pan}")
        cs.event_string(f"i 1 0 {self.noteLength} {self.tableNum} {startPitch} {endPitch} {self.amp} {self.pan}")

    def playRandomRow(self):
        row, rowNum = selectRandomRow()
        waveform = fftRow(row)
        self.fillTable(waveform)
        self.playNote()
        
        # Add row to active rows for visualization
        with display_lock:
            active_rows[rowNum] = (time.time(), self.noteLength, self.tableNum)

def createVoices(count, minNoteLen, maxNoteLen, overlap):
    cs.event_string(f"i99 0 0 {count}")
    instances = []
    noteLenStep = (maxNoteLen - minNoteLen) / (count - 1)
    amp = 1 / count

    # Create equidistant pan values (0 to 1)
    panValues = [i / (count - 1) for i in range(count)] if count > 1 else [0.5]
    # Shuffle the pan values randomly
    random.shuffle(panValues)

    for i in range(count):
        instances.append(voice(i + 1, minNoteLen + (noteLenStep * i), amp, panValues[i], overlap))
    return instances

voices = createVoices(30, 8, 15, 0.5)

for voice in voices:
    voice.checkTimeAndPlay()

print("imgWavetable running... Press 'Esc' to exit")
print("Yellow lines show currently playing rows")

while 1:
    for voice in voices:
        voice.checkTimeAndPlay()

    # break out if escape key is pressed
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

# Cleanup
cv.destroyAllWindows()
csThread.join()
sys.exit()