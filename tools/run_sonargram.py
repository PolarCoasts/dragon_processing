"""Run tools/aris_sonargram.py with the parameters below.

Edit the values below, then run:
    python tools/run_sonargram.py

Any parameter left as None is not passed, so aris_sonargram.py uses its own default.
"""

import subprocess
import sys
from pathlib import Path

# ===== Files =====
nc_file = Path('/path/to/dragon_data/proc/dragon/YYYYMMDD_HHMM_dragon/dragon_YYYYMMDD_HHMM_aris.nc')
output_dir = Path('/path/to/dragon_data/vis/YYYYMMDD_HHMM/sonargram')

# ===== Frames (use a frame index or a time for each end, not both) =====
start = 0 # First frame to export (default: 0)
end = None # Last frame to export (default: last frame in the file)
start_time = None # e.g. "2026-08-17T18:00:37", the frame nearest this time (UTC)
end_time = None # e.g. "2026-08-17T18:09:21", the frame nearest this time (UTC)

# ===== Filters =====
threshold = 50 # Intensity a sample must exceed to survive the filter, 0-254 (default: 50)
min_range = None # Closest range from the sonar (m) a sample must be at (default: no limit)
max_range = None # Farthest range from the sonar (m) a sample must be at (default: no limit)

# ===== Peak spline =====
spline = True # False to turn the peak spline off
spline_lam = None # Smoothness, larger is smoother, must be > 0 (default: chosen automatically for each segment)
spline_min_peaks = 5 # Fewest peaks a segment needs to have a spline fit through it, at least 5 (default: 5)
spline_max_gap = 3 # Most beams in a row without a peak before a segment is split (default: 3)

# ===== Outlier filter ===== #
outlier_filter = True # Outlier filter on (true) / off (false)
outlier_mad = 3 # Scales the median absolute deviation above the median score to determine outlier peaks to be removed, lower removes more (default: 3)
outlier_neighbors = 4 # Neighborhood size / number of nearest peaks each peak is compared with (default: 4)
outlier_min_diff = 0.2 # Smallest median range difference (m) from its neighbors a peak must have to be removed (default: 0.2)
outlier_max_from_median = None # Remove peaks more than this range (m) from the median range of the peaks, works even with outlier_filter = False

# ===== Structure test (whether the spline is drawn) =====
structure_test = True # False to draw the spline without testing whether the structure of peak points
structure_max_coherence = 0.5 # Low pass threshold coherence ratio for the spline to be drawn (default: 0.5) (1.0 = noise-like, 0.0 = neighboring peaks agree)
structure_max_concentration = None # Low pass threshold concentration ratio for the spline to be drawn (1 = spread like noise, 0 = all at one range)
structure_neighbors = 4 # Neighborhood size = number of nearest peaks each peak is compared with for coherence (default: 4)
structure_min_peaks = 10 # Fewest peaks a frame needs to be tested, frames with fewer get no spline (default: 10)


if start is not None and start_time is not None:
    sys.exit('Set start or start_time, not both')
if end is not None and end_time is not None:
    sys.exit('Set end or end_time, not both')

script = Path(__file__).resolve().parents[1] / 'tools' / 'aris_sonargram.py' # Found relative to this file, so it works from any directory
command = [sys.executable, str(script), str(nc_file), str(output_dir)]

options = {
    '--start': start,
    '--end': end,
    '--start-time': start_time,
    '--end-time': end_time,
    '--threshold': threshold,
    '--min-range': min_range,
    '--max-range': max_range,
    '--spline-lam': spline_lam,
    '--spline-min-peaks': spline_min_peaks,
    '--spline-max-gap': spline_max_gap,
    '--outlier-mad': outlier_mad,
    '--outlier-neighbors': outlier_neighbors,
    '--outlier-min-diff': outlier_min_diff,
    '--outlier-max-from-median': outlier_max_from_median,
    '--structure-max-coherence': structure_max_coherence,
    '--structure-max-concentration': structure_max_concentration,
    '--structure-neighbors': structure_neighbors,
    '--structure-min-peaks': structure_min_peaks,
}
for flag, value in options.items():
    if value is not None:
        command += [flag, str(value)]
if not spline:
    command.append('--no-spline')
if not outlier_filter:
    command.append('--no-outlier-filter')
if not structure_test:
    command.append('--no-structure-test')

subprocess.run(command, check=True)
