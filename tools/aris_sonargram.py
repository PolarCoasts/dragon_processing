"""Export ARIS frames from a combined *_aris.nc file as sonar fan images (.png).

Intensity is colored with viridis (0-255), and every raw sample is drawn. Filters only apply to finding the peaks:
    --threshold: samples at or below this intensity can't be a peak
    --min-range/--max-range: samples outside this distance from the sonar can't be a peak

The strongest echo along each beam that survives both filters is that beam's peak. Peaks that are outliers are removed by
a median absolute deviation (MAD) filter on each peak's median range difference to its neighbors and, if given, by a cutoff
on distance from the median range of the peaks.
A smoothing spline of range against beam angle is fit through the remaining peaks and drawn in red. Peaks are split into
segments wherever more than --spline-max-gap beams in a row have no peak, and each segment is fit separately so the
curve never bridges a gap. Segments with fewer than --spline-min-peaks peaks are not drawn.
The spline is only drawn on frames whose peaks pass a structure test, which compares the coherence and concentration of
the peaks, before outlier removal, to what pure noise across the window would give.
The CTD depth of each frame is reported in a box in the lower left of the fan.

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
    --no-spline: don't fit or draw the peak spline
    --outlier-mad: scaled MADs above the median score a peak must be to be removed, lower removes more
    --outlier-neighbors: number of nearest peaks each peak is compared with for the MAD filter
    --outlier-min-diff: smallest median range difference (m) from its neighbors a peak must have to be removed
    --no-outlier-filter: don't apply the MAD filter
    --outlier-max-from-median: remove peaks more than this range (m) from the median range of the peaks
    --structure-max-coherence: highest coherence (1.0 = noise-like spread, 0.0 = closely packed) a frame can have for the spline to be drawn
    --structure-max-concentration: highest concentration (1.0 = noise-like spread, 0.0 = all at one range) a frame can have for the spline to be drawn
    --structure-neighbors: number of nearest peaks each peak is compared with for the coherence test
    --structure-min-peaks: fewest peaks a frame needs to be tested, frames with fewer get no spline
    --no-structure-test: draw the spline without testing whether the peaks are noise-like

Usage:
    python tools/aris_sonargram.py "/Volumes/LeConte Pro/dragon_data/raw/20260817_1800_dragon/2026-08-17_aris.nc" frames/
    python tools/aris_sonargram.py <nc_file> <output_dir> --start 1000 --end 2000
    python tools/aris_sonargram.py <nc_file> <output_dir> --start-time 2026-08-17T20:15:00 --end-time 2026-08-17T20:20:00
    python tools/aris_sonargram.py <nc_file> <output_dir> --start 1000 --end 2000 --threshold 80 --min-range 2 --max-range 8
    python tools/aris_sonargram.py <nc_file> <output_dir> --threshold 80 --spline-lam 50 --spline-max-gap 5
    python tools/aris_sonargram.py <nc_file> <output_dir> --threshold 80 --outlier-mad 4 --outlier-max-from-median 2 --structure-max-concentration 0.3
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


def fit_peak_spline(peak_beams, peak_ranges, beam_angle, lam, min_peaks, max_gap):
    # Fit a smoothing spline of range against beam angle through the beam peaks, one per segment of nearby peaks, and return the curves in cartesian coordinates (meters), with NaNs between segments so they draw as separate lines

    breaks = np.flatnonzero(np.diff(peak_beams) > max_gap + 1) + 1 # Split wherever more than max_gap beams in a row have no peak
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
    mad = 1.4826 * np.median(np.abs(score - median)) # Get median of absolute deviations from center, then scale to standard deviation units to get median absolute deviation
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
    concentration = (q3 - q1) / (window_width / 2) # Concentration is the ratio of middle half of peak ranges to the interquartile range of noise (1/2 window width)
    coherence = np.median(neighbor_scores(peak_beams, peak_ranges, n_neighbors)) / (0.3 * window_width) # Ratio of median difference scores in neighborhood to that of noise, simulated to give a score of ~0.3 of the window width
    return concentration, coherence # Return both ratios

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
    end_group.add_argument('--end-time', help='Last frame to export is the one nearest this time, e.g. 2026-08-17T20:20:00')
    parser.add_argument('--threshold', type=int, default=50, help='Intensity a sample must exceed to survive the filter (default: 50)')
    parser.add_argument('--min-range', type=float, default=0, help='Closest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--max-range', type=float, default=float('inf'), help='Farthest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--spline-lam', type=float, help='Smoothness of the peak spline, larger is smoother (default: chosen automatically for each segment)')
    parser.add_argument('--spline-min-peaks', type=int, default=5, help='Fewest peaks a segment needs to have a spline fit through it (default: 5, minimum: 5)')
    parser.add_argument('--spline-max-gap', type=int, default=3, help='Most beams in a row without a peak that a segment can span before it is split (default: 3)')
    parser.add_argument('--no-spline', action='store_true', help="Don't fit or draw the peak spline")
    parser.add_argument('--outlier-mad', type=float, default=3.0, help='Scales the amount that the median absolute deviation above the median of all points in a frame a peak must be to be removed as an outlier, lower removes more (default: 3)')
    parser.add_argument('--outlier-neighbors', type=int, default=4, help='Neighborhood size / number of nearest peaks each peak is compared with for outlier removal by scaled median absolute deviation (default: 4)')
    parser.add_argument('--outlier-min-diff', type=float, default=0.2, help='Smallest median range difference (m) from its neighbors a peak must have to be removed (default: 0.2)')
    parser.add_argument('--no-outlier-filter', action='store_true', help="Don't remove outlier peaks")
    parser.add_argument('--outlier-max-from-median', type=float, help='Remove peaks more than this range (m) from the median range of the peaks, applied alongside the MAD filter (default: off)')
    parser.add_argument('--structure-max-coherence', type=float, default=0.5, help='Highest coherence (1 = noise-like spread, 0 = closely packed) a frame can have for the spline to be drawn (default: 0.5)')
    parser.add_argument('--structure-max-concentration', type=float, help='Highest concentration (1 = spread like noise, 0 = all at one range) a frame can have for the spline to be drawn (default: off)')
    parser.add_argument('--structure-neighbors', type=int, default=4, help='Neighborhood size / number of nearest peaks each peak is compared with for coherence test (default: 4)')
    parser.add_argument('--structure-min-peaks', type=int, default=10, help='Fewest peaks a frame needs to be tested, frames with fewer get no spline (default: 10)')
    parser.add_argument('--no-structure-test', action='store_true', help="Draw the spline without testing whether the peaks are noise-like")
    args = parser.parse_args()
    if not 0 <= args.threshold < 255:
        parser.error('--threshold must be between 0 and 254')
    if args.min_range >= args.max_range:
        parser.error('--min-range must be less than --max-range')
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

    # Generate figure with one fixed extent covering every selected frame, so the scale doesn't jump between recordings
    r_min, r_max = min_range[frames].min(), max_range[frames].max() # Get minimum and maximum range
    max_angle = np.deg2rad(np.abs(beam_angle).max()) # Determine extent of furthest angle
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
    ax.set_xlim(-r_max * np.sin(max_angle), r_max * np.sin(max_angle)) # Set x-axis limits
    ax.set_ylim(r_min * np.cos(max_angle), r_max) # Set y-axis limits
    ax.set_aspect('equal') # Set equal aspect
    ax.set_xlabel('Across-beam Distance (m)') # Set x-axis label
    ax.set_ylabel('Range from Sonar (m)') # Set y-axis label
    ax.set_facecolor('white') # Background of plot area
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 255), 'viridis'), ax=ax, shrink=0.7, label='Echo Intensity (0 - 255)') # Generate colorbar
    spline_line = None if args.no_spline else ax.plot([], [], color='red', linewidth=1.5, zorder=4)[0] # Spline through the beam peaks, points filled in per frame
    depth_box = ax.text(0.02, 0.02, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=5) # Report depth

    mesh, window = None, None # Initialize figure
    for i in frames: # For each frame...
        image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Get frame, distance, and bins
        n_bins = in_range.size # Set size based on number of bins
        peak_bins, peak_beams = beam_peaks(image, in_range, args.threshold) # Get strongest surviving echo on each beam

        # Sonar fan
        frame_window = (min_range[i], max_range[i], n_bins) # Establish window from frame parameters
        if frame_window == window: # If geometry has been set by the previous frame...
            mesh.set_array(image) # Same fan geometry as the last frame, update pixels only
        else: # If geometry has not been set by the previous frame
            if mesh is not None: # And the mesh has been set...
                mesh.remove() # Remove the previous mesh
            x, y = fan_coordinates(min_range[i], max_range[i], n_bins, beam_angle) # Get cartesian coordinates for each cell from sonar polar
            mesh = ax.pcolormesh(x, y, image, cmap='viridis', vmin=0, vmax=255, shading='flat') # Set mesh
            window = frame_window # Set window for next frame to reference geometry
        peak_r = ranges[peak_bins] # Range of each beam peak
        keep = np.ones(peak_beams.size, dtype=bool) # Initialize array of peaks to keep
        if not args.no_outlier_filter: # If the MAD filter is on...
            keep = remove_outliers(peak_beams, peak_r, args.outlier_neighbors, args.outlier_mad, args.outlier_min_diff) # Flag outlier peaks
        if args.outlier_max_from_median is not None: # If the median filter is on...
            keep[keep] = remove_far_from_median(peak_r[keep], args.outlier_max_from_median) # Remove peaks that are beyond the hard threshold distance from the median of the frame
        window_width = min(max_range[i], args.max_range) - max(min_range[i], args.min_range) # Get width (m) of the window peaks can fall in, the instrument's range cut down by the range filter
        concentration, coherence = peak_structure(peak_beams, peak_r, window_width, args.structure_neighbors, args.structure_min_peaks) # Get concentration and coherence ratios
        if args.no_structure_test: # If the coherence/concentration structure test is off...
            skip_reason = None # Draw a spline for all frames with peaks above minimum
        elif np.isnan(coherence): # If there are not enough peaks to test...
            skip_reason = 'too few peaks' # Report too few peaks
        elif coherence > args.structure_max_coherence: #  If the calculated coherence is greater than the user set cutoff...
            skip_reason = 'coherence' # Report spline not drawn due to coherence test failing
        elif args.structure_max_concentration is not None and concentration > args.structure_max_concentration: # If calculated concentration is greater than the user set cutoff...
            skip_reason = 'concentration' # Report spline is not drawn due to concentration test failing
        else: # If all tests are passed
            skip_reason = None # Report nothing
        if spline_line is not None: # If the spline is turned on...
            if skip_reason is None: # And if the frame passed the test...
                spline_line.set_data(*fit_peak_spline(peak_beams[keep], peak_r[keep], beam_angle, args.spline_lam, args.spline_min_peaks, args.spline_max_gap)) # Fit this frame's peaks and move the line to it
            else: # If the frame failed the test...
                spline_line.set_data([], []) # Skip the previous frame's spline
        if depth is None: # If the file has no depth variable...
            depth_box.set_text('Depth: not in file') # Report no depth available in NetCDF dataset
        elif np.isnan(depth[i]): # If depth is missing for frame
            depth_box.set_text('Depth: no CTD data') # Report no CTD data
        else: # Otherwise...
            depth_box.set_text(f'Depth: {depth[i]:.1f} m') # Report depth

        ax.set_title(
            f'frame {i} of {times.size - 1}    {str(times[i])[:23]}\n'
            f'{min_range[i]:.1f}-{max_range[i]:.1f} m, {n_bins} bins, {frame_rate[i]:.1f} FPS',
            fontsize=10,
        ) # Set title of figure
        fig.savefig(args.output_dir / f'frame_{i:0{digits}d}.png', dpi=150) # Save frame
        if (i - start) % 100 == 0:
            print(f'Frame {i}/{end}')

    plt.close(fig)
    print(f'Saved {len(frames)} frames to {args.output_dir}')

# =============================

if __name__ == '__main__':
    main()
