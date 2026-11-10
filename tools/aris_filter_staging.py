"""Export ARIS frames as .png images showing the sonargram next to that frame's intensity and range histograms.

Samples at or below --threshold are filtered out and drawn blue, dark at 0 to light at the threshold. Samples that
survive are drawn red, light just above the threshold to dark at 255. Neither side reaches white, so there is a hard cut at the threshold.
Samples outside --min-range/--max-range are filtered out entirely: they are gray in the fan and
left out of the intensity histogram.
The strongest echo along each beam that survives both filters is marked yellow in the fan.
A smoothing spline of range against beam angle is fit through those peaks and drawn in cyan. Peaks are split into
segments wherever more than --spline-max-gap beams in a row have no peak, and each segment is fit separately so the
curve never bridges a gap. Segments with fewer than --spline-min-peaks peaks are not drawn.

The range histogram shows how many samples survive the intensity filter at each distance from the sonar,
with the range filter cutoffs drawn as dashed lines.

Arguments:
    <nc_file>: input NetCDF file with dragon aris3000 data
    <output_dir>: directory to which rendered frames will be written
    --start/--end: start and end frames for visualization
    --start-time/--end-time: start and end datetime timestamp for visualization (UTC), will approximate to closest timestamp
    --min-range/--max-range: specifies minimum and maximum extents for band-pass range filter
    --threshold: specifies high-pass intensity threshold value
    --spline-lam: smoothness of the peak spline, larger is smoother (default: chosen automatically for each segment)
    --spline-min-peaks: fewest peaks a segment needs to have a spline fit through it (at least 5)
    --spline-max-gap: most beams in a row without a peak that a segment can span before it is split
    --spline-max-jump: largest range difference (m) between consecutive peaks before a segment is split (default: off)
    --no-spline: don't fit or draw the peak spline

Usage:
    python tools/aris_filter_staging.py <nc_file> <output_dir>
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start 1000 --end 2000
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start-time 2026-08-17T20:15:00 --end-time 2026-08-17T20:20:00
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start 1000 --end 2000 --threshold 80
    python tools/aris_filter_staging.py <nc_file> <output_dir> --threshold 80 --min-range 2 --max-range 8
    python tools/aris_filter_staging.py <nc_file> <output_dir> --threshold 80 --spline-lam 50 --spline-max-gap 5
"""

# ===== Dependencies =====
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # No window is opened, images are only saved
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_smoothing_spline
import xarray as xr
# ========================

# ===== Helper Functions =====
def edges(centers):
    # Get the edges of each cell, halfway between the center of current cell and the next

    mid = (centers[1:] + centers[:-1]) / 2 # Find midpoint between current cell and next
    return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]]) # Generate the first and last edges, and return that with all other midpoints


def fan_coordinates(min_range, max_range, n_bins, beam_angle):
    # Get list of all cell corners in cartesian coordinates for pcolormesh

    ranges = np.linspace(min_range, max_range, n_bins) # Get range extents
    angles = np.deg2rad(beam_angle) # Get per-beam angles
    r, a = np.meshgrid(edges(ranges), edges(angles), indexing='ij') # Generate a list of cell corners from range (arcs some distance from sonar) and angle (straight rays propagating from sonar at some angle) edges
    return r * np.sin(a), r * np.cos(a) # Convert from polar to cartesian (meters)


def nearest_frame(times, timestamp):
    # Get the index of the frame whose time is closest to the timestamp (e.g. 2026-08-17T20:15:00)

    return int(np.abs(times - np.datetime64(timestamp)).argmin()) # Subtract specified timestamp from all timestamps and get lowest


def read_frame(ds, i, filter_min, filter_max):
    # Get frame i, the distance of each of its bins from the sonar, and which bins fall inside the range filter

    n_bins = int(ds.number_of_bins[i]) # Get number of bins
    image = ds.frame[i, :n_bins, :].values # Drop the zero padding beyond this frame's bins
    ranges = np.linspace(float(ds.min_range[i]), float(ds.max_range[i]), n_bins) # Distance of each bin from the sonar
    in_range = (ranges >= filter_min) & (ranges <= filter_max) # Bins between the range filter cutoffs
    return image, ranges, in_range


def beam_peaks(image, in_range, threshold):
    # Get the bin of the strongest echo along each beam, only counting samples inside the range filter that exceed the threshold

    candidates = np.where(in_range[:, np.newaxis], image, 0) # Zero out bins outside the range filter so they can't be the peak
    peak_bins = candidates.argmax(axis=0) # Bin of the maximum intensity along each beam (nearest bin wins ties)
    beams = np.arange(image.shape[1]) # Index of every beam
    has_peak = candidates[peak_bins, beams] > threshold # Beams whose strongest echo survives the intensity filter
    return peak_bins[has_peak], beams[has_peak]


def threshold_cmap(threshold):
    # Get the blue (filtered out) / red (survives) colormap with a hard cut at the threshold

    lightest = 0.6 # Lightest extent of blue and red symbology for under/over intensity cutoff threshold
    filtered_colors = plt.cm.Blues(np.linspace(1, lightest, threshold + 1)) # Set filtered intensities to blue
    survivor_colors = plt.cm.Reds(np.linspace(lightest, 1, 255 - threshold)) # Set survivor intensities to red
    return matplotlib.colors.ListedColormap(np.vstack([filtered_colors, survivor_colors]), name='filtered_blue_survivor_red').with_extremes(bad='gray') # Initialize color map


def fit_peak_spline(peak_beams, peak_ranges, beam_angle, lam, min_peaks, max_gap, max_jump):
    # Fit a smoothing spline of range against beam angle through the beam peaks, one per segment of nearby peaks, and return the curves in cartesian coordinates (meters), with NaNs between segments so they draw as separate lines

    split = np.diff(peak_beams) > max_gap + 1 # Split wherever more than max_gap beams in a row have no peak
    if max_jump is not None: # If the range jump split is on...
        split |= np.abs(np.diff(peak_ranges)) > max_jump # Split wherever consecutive peaks are more than max_jump meters apart in range
    breaks = np.flatnonzero(split) + 1 # Index of the first peak of each new segment

    x, y = [], []
    for segment in np.split(np.arange(peak_beams.size), breaks): # For each segment...
        if segment.size < min_peaks: # Too few peaks to fit a meaningful spline, skip it
            continue
        angles, ranges = beam_angle[peak_beams[segment]], peak_ranges[segment] # Angle and range of each peak in the segment
        order = np.argsort(angles) # Spline needs strictly increasing angles
        spline = make_smoothing_spline(angles[order], ranges[order], lam=lam) # Make spline, lam sets smoothness, no argument sets automatically
        fit_angles = np.linspace(angles.min(), angles.max(), 200) # Evaluate only between the outermost peaks, never extrapolate
        fit_ranges = spline(fit_angles)
        x += [fit_ranges * np.sin(np.deg2rad(fit_angles)), [np.nan]] # Convert from polar to cartesian (meters), NaN breaks the line
        y += [fit_ranges * np.cos(np.deg2rad(fit_angles)), [np.nan]]
    if not x: # No segment had enough peaks
        return np.empty(0), np.empty(0)
    return np.concatenate(x), np.concatenate(y)


def neighbor_scores(peak_beams, peak_ranges, n_neighbors):
    # Get each peak's median range difference (m) to its n number of user defined neighboring peaks by beam number

    beam_dist = np.abs(peak_beams[:, np.newaxis] - peak_beams[np.newaxis, :]) # Create a matrix of beam distances from one another
    np.fill_diagonal(beam_dist, beam_dist.max() + 1) # Prevent a peak from being its own neighbor by giving the matrix cell pairing a beam with itself a distance which is beyond the max number of beams
    neighbors = np.argsort(beam_dist, axis=1, kind='stable')[:, :n_neighbors] # Indices of the nearest peaks to each peak, ties go to lower beam
    return np.median(np.abs(peak_ranges[:, np.newaxis] - peak_ranges[neighbors]), axis=1) # Median range difference (m) to ranges of neighbors


def remove_outliers(peak_beams, peak_ranges, n_neighbors, n_mad, min_diff):
    # Score each peak by the median range difference to its nearest peaks, and flag peaks whose score is greater than the median range difference of the neighborhood

    keep = np.ones(peak_beams.size, dtype=bool) # Initialize array of peaks to keep
    if peak_beams.size <= n_neighbors: # If there are too few peaks for a neighborhood size of n...
        return keep # Keep them all
    score = neighbor_scores(peak_beams, peak_ranges, n_neighbors) # Get the median range difference score of each neighbor
    median = np.median(score) # Find the median of median range difference scores for the neighborhood
    mad = 1.4826 * np.median(np.abs(score - median)) # Get median of absolute deviations from center, then scale to standard deviation units to get median abolsute deviation
    return ~((score > median + n_mad * mad) & (score > min_diff)) # Return a mask of the peaks to keep


def remove_far_from_median(peak_ranges, max_dist):
    # Flag peaks whose range is more than some user specified distance in meters from the median range of all the peaks

    if peak_ranges.size == 0: # If there are no peaks...
        return np.ones(0, dtype=bool) # Return nothing
    return np.abs(peak_ranges - np.median(peak_ranges)) <= max_dist # Return only those with a distance in meters below that specified


def peak_structure(peak_beams, peak_ranges, window_width, n_neighbors, min_peaks):
    # Get the concentration (how tightly the peak ranges cluster) and coherence (how well neighboring peaks agree) of a frame's peak each divided by what pure noise across the window would give, values nearer to 1 have noise-like spread and values nearer to 0 are closely packed

    if peak_beams.size < min_peaks or window_width <= 0: # If there are too few peaks
        return np.nan, np.nan # Return NaN
    q1, q3 = np.percentile(peak_ranges, [25, 75]) # Retrieve value that represent the extents of band holding the middle half of the peak ranges (1st and 3rd quartiles)
    concentration = (q3 - q1) / (window_width / 2) # Concentration is the of middle half of peak ranges to the interquartile range of noise (1/2 window width)
    coherence = np.median(neighbor_scores(peak_beams, peak_ranges, n_neighbors)) / (0.3 * window_width) # Ratio of median difference scores in neighborhood to that of noise, simulated to give a score of ~0.3 of the window width
    return concentration, coherence # Return both ratios


def structure_skip_reason(concentration, coherence, max_coherence, max_concentration):
    # Get why a frame fails the structure test, None if it passes

    if np.isnan(coherence): # If there are not enough peaks to test...
        return 'too few peaks' # Report too few peaks
    if coherence > max_coherence: #  If the calculated coherence is greater than the user set cutoff...
        return 'coherence' # Report spline not drawn due to coherence test failing
    if max_concentration is not None and concentration > max_concentration: # If calculated concentration is greater than the user set cutoff...
        return 'concentration' # Report spline is not drawn due to concentration test failing
    return None # If all tests are passed, report nothing

# ============================

# ===== Main Runtime Loop =====
def main():

    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('nc_file', type=Path)
    parser.add_argument('output_dir', type=Path)
    start_group = parser.add_mutually_exclusive_group() # Start by frame index or by time, not both
    start_group.add_argument('--start', type=int, default=0, help='First frame to export (default: 0)')
    start_group.add_argument('--start-time', help='First frame to export is the one nearest this time, e.g. 2026-08-17T20:15:00')
    end_group = parser.add_mutually_exclusive_group() # End by frame index or by time, not both
    end_group.add_argument('--end', type=int, help='Last frame to export (default: last frame in the file)')
    end_group.add_argument('--end-time', help=':ast frame to export is the one nearest this time, e.g. 2026-08-17T20:20:00')
    parser.add_argument('--threshold', type=int, default=50, help='Intensity a sample must exceed to survive the filter (default: 50)')
    parser.add_argument('--dynamic-threshold-coef', type=float, default=0.0, help="Intensity the threshold is lowered for every meter a frame's median peak range is beyond the sonar's minimum range for that frame, --threshold applies at the sonar's minimum range, must be 0 or more (default: 0, dynamic threshold off)")
    parser.add_argument('--dynamic-threshold-hold', type=int, default=10, help='Most frames since the last frame that passed the structure test that a frame failing the coherence or concentration test can keep that frame\'s threshold, after which it resets to --threshold (default: 10)')
    parser.add_argument('--min-range', type=float, default=0, help='Closest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--max-range', type=float, default=float('inf'), help='Farthest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--range-bins', type=int, default=100, help='Number of bins in the range histogram (default 100)')
    parser.add_argument('--spline-lam', type=float, help='Smoothness of the peak spline, larger is smoother (default: chosen automatically for each segment)')
    parser.add_argument('--spline-min-peaks', type=int, default=5, help='Fewest peaks a segment needs to have a spline fit through it (default: 5, minimum: 5)')
    parser.add_argument('--spline-max-gap', type=int, default=3, help='Most beams in a row without a peak that a segment can span before it is split (default: 3)')
    parser.add_argument('--spline-max-jump', type=float, help='Largest range difference (m) between consecutive peaks before a segment is split (default: off)')
    parser.add_argument('--no-spline', action='store_true', help="Don't fit or draw the peak spline")
    parser.add_argument('--outlier-mad', type=float, default=3.0, help='Scales the amount that the median absolute deviation above the median of all points in a frame a peak must be to be removed as an outlier, lower removes more (default: 3)')
    parser.add_argument('--outlier-neighbors', type=int, default=4, help='Neighborhood size / number of nearest peaks each peak is compared with for outlier removal by scaled median absolute deviation (default: 4)')
    parser.add_argument('--outlier-min-diff', type=float, default=0.2, help='Smallest median range difference (m) from its neighbors a peak must have to be removed (default: 0.2)')
    parser.add_argument('--no-outlier-filter', action='store_true', help="Don't remove outlier peaks")
    parser.add_argument('--outlier-max-from-median', type=float, help='Remove peaks more than this range (m) from the median range of the peaks, applied alongside the MAD filter (default: off)')
    parser.add_argument('--structure-max-coherence', type=float, default=0.5, help='Highest coherence (1.0 = noise-like spread, 0.0 = closely packed) a frame can have for the spline to be drawn (default: 0.5)')
    parser.add_argument('--structure-max-concentration', type=float, help='Highest concentration (1.0 = noise-like spread, 0.0 = all at one range) a frame can have for the spline to be drawn (default: off)')
    parser.add_argument('--structure-neighbors', type=int, default=4, help='Neighborhood size / number of nearest peaks each peak is compared with for coherence test (default: 4)')
    parser.add_argument('--structure-min-peaks', type=int, default=10, help='Fewest peaks a frame needs to be tested, frames with fewer get no spline (default: 10)')
    parser.add_argument('--no-structure-test', action='store_true', help="Draw the spline without testing whether the peaks are noise-like")
    args = parser.parse_args()
    if not 0 <= args.threshold < 255:
        parser.error('--threshold must be between 0 and 254')
    if args.dynamic_threshold_coef < 0:
        parser.error('--dynamic-threshold-coef must be 0 or more')
    if args.dynamic_threshold_hold < 0:
        parser.error('--dynamic-threshold-hold must be 0 or more')
    if args.min_range >= args.max_range:
        parser.error('--min-range must be less than --max-range')
    if args.spline_max_jump is not None and args.spline_max_jump <= 0:
        parser.error('--spline-max-jump must be greater than 0')
    if args.spline_lam is not None and args.spline_lam <= 0:
        parser.error('--spline-lam must be greater than 0')
    if args.spline_min_peaks < 5:
        parser.error('--spline-min-peaks must be at least 5')
    if args.spline_max_gap < 0:
        parser.error('--spline-max-gap must be 0 or more')
    if args.outlier_mad < 0: 
        parser.error('--outlier-mad must be 0 or more')
    if args.outlier_neighbors < 1:
        parser.error('--outlier-neighbors must be at least 1')
    if args.outlier_min_diff < 0:
        parser.error('--outlier-min-diff must be 0 or more') 
    if args.outlier_max_from_median is not None and args.outlier_max_from_median <= 0:
        parser.error('--outlier-max-from-median must be greater than 0')
    if args.structure_max_coherence <= 0: 
        parser.error('--structure-max-coherence must be greater than 0')
    if args.structure_max_concentration is not None and args.structure_max_concentration <= 0:
        parser.error('--structure-max-concentration must be greater than 0')
    if args.structure_neighbors < 1:
        parser.error('--structure-neighbors must be at least 1')
    if args.structure_min_peaks <= args.structure_neighbors:
        parser.error('--structure-min-peaks must be greater than --structure-neighbors')

    # Load dataset (frames stay on disk and are read one at a time)
    ds = xr.open_dataset(args.nc_file)

    # Get pertinent values from dataset
    times = ds.time.values
    min_range = ds.min_range.values
    max_range = ds.max_range.values
    frame_rate = ds.frame_rate.values
    beam_angle = ds.beam_angle.values
    depth = ds.depth.values if 'depth' in ds else None

    # Start at specified frame if no start timestamp is specified, or else start the frame nearest to specified start time
    start = args.start if args.start_time is None else nearest_frame(times, args.start_time)
    if args.end_time is not None: # If end time is specified...
        end = nearest_frame(times, args.end_time) # End at frame with timestamp nearest to that specified
    else: # If no end time is specified...
        end = times.size - 1 if args.end is None else min(args.end, times.size - 1) # Check if end frame is specified, if not pick last frame
    if start > end: # Throw an error if the start time is later than the end time
        parser.error(f'start frame ({start}) is after end frame ({end})')
    frames = range(start, end + 1)
    print(f'exporting frames {start}-{end} ({str(times[start])[:23]} to {str(times[end])[:23]})')

    args.output_dir.mkdir(parents=True, exist_ok=True) # Create the output directory if needed
    digits = len(str(times.size - 1)) # Zero-pad file names so they sort in frame order

    # Fixed range bin edges covering every selected frame, so the range axis doesn't jump between recordings
    r_min, r_max = min_range[frames].min(), max_range[frames].max() # Get minimum and maximum range
    range_edges = np.linspace(r_min, r_max, args.range_bins + 1) # Get edges of these for visualization

    # Dynamic threshold, lowered for each frame by coef for every meter its median peak range is beyond the sonar's minimum range for that frame
    thresholds = np.full(len(frames), args.threshold, dtype=int) # Intensity threshold of each frame, the base threshold unless the dynamic threshold is on
    if args.dynamic_threshold_coef > 0: # If the dynamic threshold is on...
        median_ranges = np.full(len(frames), np.nan) # Median peak range of each frame, NaN for frames with no peaks or that fail the structure test
        n_carried = 0 # Number of frames that kept the previous frame's threshold
        n_held = 0 # Number of noise-like frames that kept the last passing frame's threshold
        last_pass = args.threshold # Threshold of the last frame that passed the structure test
        since_pass = 0 # Frames since the last frame that passed the structure test
        for k, i in enumerate(frames): # For each frame, in order, so each one can use the threshold set for the one before it...
            image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Read the frame
            previous = thresholds[k - 1] if k > 0 else args.threshold # Previous frame's threshold, the base threshold for the first frame
            since_pass += 1 # Count this frame, reset below if it passes
            peak_bins, peak_beams = beam_peaks(image, in_range, previous) # First pass peaks, found with the previous frame's threshold
            if peak_bins.size == 0: # If there are no peaks to take a median of...
                thresholds[k] = previous # Keep the previous frame's threshold instead of snapping back to the base threshold
                n_carried += 1
                continue # Leave it NaN
            peak_r = ranges[peak_bins] # Range of each beam peak
            if not args.no_structure_test: # If the structure test is on...
                window_width = min(max_range[i], args.max_range) - max(min_range[i], args.min_range) # Get width (m) of the window peaks can fall in
                concentration, coherence = peak_structure(peak_beams, peak_r, window_width, args.structure_neighbors, args.structure_min_peaks) # Get concentration and coherence ratios
                skip_reason = structure_skip_reason(concentration, coherence, args.structure_max_coherence, args.structure_max_concentration) # Get why the frame fails, None if it passes
                if skip_reason == 'too few peaks': # If the frame fails only for having too few peaks...
                    thresholds[k] = previous # Keep the previous frame's threshold instead of snapping back to the base threshold
                    n_carried += 1
                    continue # Leave it NaN
                if skip_reason is not None: # If the frame fails the coherence or concentration test, its peaks look like noise...
                    if since_pass <= args.dynamic_threshold_hold: # And the last passing frame is recent...
                        thresholds[k] = last_pass # Keep the last passing frame's threshold
                        n_held += 1
                    continue # Leave it NaN, otherwise it resets to the base threshold
            median_ranges[k] = np.median(peak_r) # Get its median peak range
            shifted = args.threshold - args.dynamic_threshold_coef * (median_ranges[k] - min_range[i]) # Lower the base threshold linearly with how far the median peak range is beyond the sonar's minimum range, where the threshold equals --threshold
            thresholds[k] = np.clip(np.round(shifted), 0, 254) # Round to a whole intensity and keep it a valid threshold
            last_pass = thresholds[k] # Remember it for later noise-like frames
            since_pass = 0 # Restart the hold
        has_median = ~np.isnan(median_ranges) # Frames with peaks that pass the structure test
        n_reset = len(frames) - has_median.sum() - n_carried - n_held # Noise-like frames past the hold, which use the base threshold
        print(f'Dynamic threshold: {has_median.sum()} of {len(frames)} frames shifted, {n_carried} kept the previous threshold, {n_held} held the last passing threshold, {n_reset} reset, thresholds {thresholds.min()}-{thresholds.max()}')

    # Count every selected frame first, so both histograms' y-axes can be fixed to their tallest bar
    intensity_counts = np.zeros((len(frames), 256), dtype='i8') # Initialize intensity counter
    range_counts = np.zeros((len(frames), args.range_bins), dtype='i8') # Initialize range counter
    for k, i in enumerate(frames): # For each frame...
        image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Read the frame
        intensity_counts[k] = np.bincount(image[in_range].ravel(), minlength=256) # Get every intensity value inside the range filter
        survivors_per_bin = (image > thresholds[k]).sum(axis=1) # Number of beams surviving the intensity filter at each bin
        range_counts[k] = np.histogram(ranges, bins=range_edges, weights=survivors_per_bin)[0] # Get every range value 
    print(f'Counted {len(frames)} frames')

    # Set symbology
    cmap = threshold_cmap(args.threshold) # Initialize color map
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=255.5) # Map intensity values from 0 to 1

    # Generate figure: sonar fan on the left, intensity and range histograms stacked on the right
    fig, axes = plt.subplot_mosaic([['fan', 'intensity'], ['fan', 'range']], figsize=(13, 7), layout='constrained')
    ax_fan, ax_intensity, ax_range = axes['fan'], axes['intensity'], axes['range']

    # Sonar fan with one fixed extent covering every selected frame
    max_angle = np.deg2rad(np.abs(beam_angle).max()) # Determine extent of furthest angle
    ax_fan.set_xlim(-r_max * np.sin(max_angle), r_max * np.sin(max_angle)) # Set x-axis limits
    ax_fan.set_ylim(r_min * np.cos(max_angle), r_max) # Set y-axis limits
    ax_fan.set_aspect('equal') # Set equal aspect
    ax_fan.set_xlabel('Across-beam Distance (m)') # Set x-axis label
    ax_fan.set_ylabel('Range from Sonar (m)') # Set y-axis label
    ax_fan.set_facecolor('white') # Background of plot area
    colorbar_mappable = plt.cm.ScalarMappable(norm, cmap) # Kept so the colorbar can be recolored for each frame's threshold
    colorbar = fig.colorbar(colorbar_mappable, ax=ax_fan, shrink=0.7, label=f'Echo Intensity (Blue: ≤ {args.threshold}, Filtered Out; Red: Survives)') # Generate colorbar
    colorbar.set_ticks([0, args.threshold, 255]) # Sets numbers on colorbar
    threshold_marker = colorbar.ax.axhline(args.threshold + 0.5, color='black', linewidth=1) # Set line marking the intensity threshold
    peaks = ax_fan.scatter([], [], s=8, color='yellow', edgecolors='none', zorder=3, label='Beam Peak') # Marker for the strongest surviving echo on each beam, positions filled in per frame
    rejected = None if args.no_outlier_filter and args.outlier_max_from_median is None else ax_fan.scatter([], [], s=12, marker='x', color='magenta', linewidths=0.8, zorder=3, label='Rejected Peak') # Marker for peaks removed as outliers
    spline_line = None if args.no_spline else ax_fan.plot([], [], color='cyan', linewidth=1.5, zorder=4, label='Peak Spline')[0] # Spline through the beam peaks, points filled in per frame
    ax_fan.legend(loc='upper right') # Visualize legend
    peak_count = ax_fan.text(0.02, 0.98, '', transform=ax_fan.transAxes, ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5) # Report number of peaks
    depth_box = ax_fan.text(0.02, 0.02, '', transform=ax_fan.transAxes, ha='left', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5) # Report depth

    # Intensity histogram, one bar per possible uint8 value, each colored the same as in the fan
    intensities = np.arange(256) # Initialize array
    intensity_bars = ax_intensity.bar(intensities, np.zeros(256), width=1, color=cmap(norm(intensities)), edgecolor='none') # Set color
    ax_intensity.set_xlim(-0.5, 255.5) # Set limit of x-axis based on possible intensity values with some headroom
    ax_intensity.set_ylim(0, max(intensity_counts.max(), 1) * 1.05) # Set limit to be tallest bar across the selected frames with some headroom
    intensity_line = ax_intensity.axvline(args.threshold, color='black', linestyle='--', label=f'Intensity Threshold ({args.threshold})') # Threshold line marker
    ax_intensity.set_xlabel('Echo Intensity (0 - 255)') # X-axis label
    ax_intensity.set_ylabel('Number of Samples') # Y-axis label
    intensity_legend = ax_intensity.legend() # Visualize legend

    # Range histogram of samples surviving the intensity filter
    range_centers = (range_edges[1:] + range_edges[:-1]) / 2 # Get range centers again
    range_colors = np.where((range_centers >= args.min_range) & (range_centers <= args.max_range), 'red', 'gray') # Red inside the range filter, gray outside
    range_bars = ax_range.bar(range_centers, np.zeros(args.range_bins), width=np.diff(range_edges), color=range_colors, edgecolor='none') # Generate bar of number of cells with intensity over threshold at each range
    ax_range.set_xlim(r_min, r_max) # Set x-limit based on range minimum and maximum set by instrument during collection
    ax_range.set_ylim(0, max(range_counts.max(), 1) * 1.05) # Tallest bar across the selected frames, with a little headroom
    for cutoff in (args.min_range, args.max_range): # Draw cutoff lines for those given
        if r_min < cutoff < r_max: # Check that cutoffs are actually in range
            ax_range.axvline(cutoff, color='black', linestyle='--') # Draw to figure
    ax_range.set_xlabel('Range from Sonar (m)') # X-axis label
    ax_range.set_ylabel(f'Samples with Intensity > {args.threshold}') # Y-axis label

    mesh, window = None, None # Initialize figure
    for k, i in enumerate(frames): # For each frame...
        image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Get frame, distance, and bins
        n_bins = in_range.size # Set size based on number of bins
        threshold = int(thresholds[k]) # This frame's intensity threshold
        peak_bins, peak_beams = beam_peaks(image, in_range, threshold) # Get strongest surviving echo on each beam
        image = np.ma.masked_array(image, mask=np.broadcast_to(~in_range[:, np.newaxis], image.shape)) # Mask bins outside the range filter so they draw gray

        # Dynamic threshold
        if args.dynamic_threshold_coef > 0: # If the dynamic threshold is on, move everything that shows the threshold to this frame's
            cmap = threshold_cmap(threshold) # Rebuild the colormap with the cut at this frame's threshold
            if mesh is not None: # If the fan has already been drawn...
                mesh.set_cmap(cmap) # Recolor it
            colorbar_mappable.set_cmap(cmap) # Recolor colorbar
            colorbar.update_normal(colorbar_mappable) # Redraw colorbar
            colorbar.set_ticks([0, threshold, 255]) # Move threshold tick
            colorbar.set_label(f'Echo Intensity (Blue: ≤ {threshold}, Filtered Out; Red: Survives)') # Update label
            threshold_marker.set_ydata([threshold + 0.5] * 2) # Move threshold line on colorbar
            for bar, color in zip(intensity_bars, cmap(norm(intensities))): # Recolor intensity histogram
                bar.set_facecolor(color) # Recolor bar
            intensity_line.set_xdata([threshold] * 2) # Move threshold line on intensity histogram
            intensity_legend.get_texts()[0].set_text(f'Intensity Threshold ({threshold})') # Update legend
            ax_range.set_ylabel(f'Samples with Intensity > {threshold}') # Update range histogram label

        # Sonar fan
        frame_window = (min_range[i], max_range[i], n_bins) # Establish window from frame parameters
        if frame_window == window: # If geometry has been set by the previous frame...
            mesh.set_array(image) # Same fan geometry as the last frame, update pixels only
        else: # If geometry has not been set by the previous frame
            if mesh is not None: # And the mesh has been set...
                mesh.remove() # Remove the previous mesh
            x, y = fan_coordinates(min_range[i], max_range[i], n_bins, beam_angle) # Get cartesian coordinates for each cell from sonar polar
            mesh = ax_fan.pcolormesh(x, y, image, cmap=cmap, norm=norm, shading='flat') # Set mesh
            window = frame_window # Set window for next frame to reference geometry

        # Peaks
        peak_r, peak_a = ranges[peak_bins], np.deg2rad(beam_angle[peak_beams]) # Range and angle of each beam peak
        peak_x, peak_y = peak_r * np.sin(peak_a), peak_r * np.cos(peak_a) # Convert peaks from polar to cartesian (meters), split out of the line below so rejected peaks can use it too
        keep = np.ones(peak_beams.size, dtype=bool) # Intialize array of peaks to keep
        if not args.no_outlier_filter: # If the MAD filter is on...
            keep = remove_outliers(peak_beams, peak_r, args.outlier_neighbors, args.outlier_mad, args.outlier_min_diff) # Flag outlier peaks
        n_mad_rejected = (~keep).sum() # Compile array of peaks removed by the MAD filter
        if args.outlier_max_from_median is not None: # If the median filter is on...
            keep[keep] = remove_far_from_median(peak_r[keep], args.outlier_max_from_median) # Remove peaks that are beyond the hard threshold distance from the median of the frame
        if rejected is not None: # If either filter is on...
            rejected.set_offsets(np.column_stack([peak_x[~keep], peak_y[~keep]])) # Remove outlier peaks
        peaks.set_offsets(np.column_stack([peak_x[keep], peak_y[keep]])) # Move markers to this frame's peaks, converted from polar to cartesian (meters)
        count_text = f'Peaks: {keep.sum()} of {beam_angle.size} beams' # Get number of kept peaks out of the total beams
        if args.dynamic_threshold_coef > 0: # If the dynamic threshold is on...
            count_text += f'\nThreshold: {threshold}' # Report this frame's threshold
        if rejected is not None: # If either filter is on...
            count_text += f'\nRejected: {(~keep).sum()} (MAD {n_mad_rejected}, median {(~keep).sum() - n_mad_rejected})' # Report how many peaks were removed, and by which filter
        window_width = min(max_range[i], args.max_range) - max(min_range[i], args.min_range) # Get width (m) of the window peaks can fall in, the instrument's range cut down by the range filter

        # Structure Test — Concentration / Coherence
        concentration, coherence = peak_structure(peak_beams, peak_r, window_width, args.structure_neighbors, args.structure_min_peaks) # Get concentration and coherence ratios
        count_text += f'\nConcentration: {concentration:.2f}\nCoherence: {coherence:.2f}' # Report both ratios in figure
        if args.no_structure_test: # If the coherence/concentration structure test is off...
            skip_reason = None # Draw a spline for all frames with peaks above minimum
        else:
            skip_reason = structure_skip_reason(concentration, coherence, args.structure_max_coherence, args.structure_max_concentration) # Get why the frame fails, None if it passes
        if spline_line is not None: # If the spline is turned on...
            if skip_reason is None: # And if the frame passed the test...
                spline_line.set_data(*fit_peak_spline(peak_beams[keep], peak_r[keep], beam_angle, args.spline_lam, args.spline_min_peaks, args.spline_max_gap, args.spline_max_jump)) # Fit this frame's peaks and move the line to it
                count_text += '\nSpline: drawn' # Report tha the spline has been drawn
            else: # If the frame failed the test...
                spline_line.set_data([], []) # Skip the previous frame's spline
                count_text += f'\nSpline: skipped ({skip_reason})' # Report skipped spline draw and reason
        peak_count.set_text(count_text) # U\pdate the box for this frame
        if depth is None: # If the file has no depth variable...
            depth_box.set_text('Depth: not in file') # Report no depth available in NetCDF dataset
        elif np.isnan(depth[i]): # If depth is missing for frame
            depth_box.set_text('Depth: no CTD data') # Report no CTD data
        else: # Otherwise...
            depth_box.set_text(f'Depth: {depth[i]:.1f} m') # Report depth

        # Histograms
        for bar, count in zip(intensity_bars, intensity_counts[k]):
            bar.set_height(count) # Set height of each bar in intensity histogram
        for bar, count in zip(range_bars, range_counts[k]):
            bar.set_height(count) # Set height of each bar in range histogram
        fig.suptitle(
            f'frame {i} of {times.size - 1}    {str(times[i])[:23]}\n'
            f'{min_range[i]:.1f}-{max_range[i]:.1f} m, {n_bins} bins, {frame_rate[i]:.1f} FPS',
            fontsize=10,
        ) # Set title of whole figure
        fig.savefig(args.output_dir / f'frame_{i:0{digits}d}.png', dpi=120) # Save frame
        if (i - start) % 100 == 0:
            print(f'Frame {i}/{end}')

    plt.close(fig)
    print(f'Saved {len(frames)} frames to {args.output_dir}')

# =============================

if __name__ == '__main__':
    main()
