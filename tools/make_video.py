"""Run ffmpeg to compile a directory of rendered frames into an .mp4 video, with the parameters below.

Edit the values below, then run:
    python tools/make_video.py

Frames are read in file name order, so zero-padded names like frame_05000.png play in frame order.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# ===== Files =====
frames_dir = Path('/path/to/dragon_data/vis/YYYYMMDD_HHMM/filter_staging')
output_file = None # Video to write (default: <frames_dir>.mp4, next to the frames directory)
pattern = 'frame_*.png' # Which files in frames_dir are frames

# ===== Video =====
framerate = 10 # Frames per second, set to the ARIS frame rate (in each frame's title) for real time, lower to slow it down
crf = 18 # Quality, lower is sharper and larger, 0-51 (18 keeps thin lines and small markers crisp)
preset = 'medium' # Encoding speed, slower makes smaller files at the same quality: 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
width = None # Scale the video to this width in pixels, keeping the aspect ratio (default: frames' own size)

# ===== Run =====
overwrite = False # True to replace output_file if it already exists


if shutil.which('ffmpeg') is None:
    sys.exit('ffmpeg not found, install it with: brew install ffmpeg')
if not frames_dir.is_dir():
    sys.exit(f'Frames directory not found: {frames_dir}')
n_frames = len(list(frames_dir.glob(pattern)))
if n_frames == 0:
    sys.exit(f'No files matching {pattern} in {frames_dir}')
if output_file is None:
    output_file = frames_dir.with_suffix('.mp4') # e.g. .../YYYYMMDD_HHMM/filter_staging.mp4
if output_file.exists() and not overwrite:
    sys.exit(f'{output_file} already exists, set overwrite = True to replace it')

# H.264 needs even dimensions, so scale to the requested width (or keep the frames' size) and round both sides to even pixels
size_filter = f'scale={width}:-2' if width is not None else 'pad=ceil(iw/2)*2:ceil(ih/2)*2'

command = [
    'ffmpeg',
    '-y' if overwrite else '-n', # Replace or keep an existing output_file
    '-framerate', str(framerate),
    '-pattern_type', 'glob', '-i', str(frames_dir / pattern), # Every matching frame, in file name order
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p', # Needed for QuickTime, PowerPoint and browsers to play it
    '-crf', str(crf),
    '-preset', preset,
    '-vf', size_filter,
    str(output_file),
]

print(f'Compiling {n_frames} frames from {frames_dir} at {framerate} fps ({n_frames / framerate:.1f} s)')
subprocess.run(command, check=True)
print(f'Saved {output_file}')
