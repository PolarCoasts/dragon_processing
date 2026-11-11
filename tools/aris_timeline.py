"""Print basic statistics for each ARIS recording in a deployment, in time order.

Usage:
    python tools/aris_timeline.py "path/to/files"
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from dragon_processing.utilities import convert_matlab_datenum


def find_aris_mat_files(deployment_path):
    aris_paths = sorted(deployment_path.rglob('*.aris'))
    aris_paths = [
        path for path in aris_paths
        if path.parent.name != 'files_that_are_not_parsing' and not path.name.startswith('._')
    ]
    return [path.with_suffix('.mat') for path in aris_paths if path.with_suffix('.mat').is_file()]


def file_stats(mat_file, batch=200):
    with h5py.File(mat_file, 'r') as f:
        header = f['header']
        value = lambda key: np.squeeze(header[key][()])
        n_frames = int(value('numframes'))
        frame_rate = float(value('framerate'))
        start = pd.Timestamp(convert_matlab_datenum(value('datenum'))).round('ms')
        frame = f['frame']
        vmin, vmax = np.inf, -np.inf
        for i in range(0, frame.shape[-1], batch):
            block = frame[:, :, i:i+batch]
            vmin, vmax = min(vmin, block.min()), max(vmax, block.max())

        return {
            'start': start,
            'end': (start + pd.Timedelta(seconds=n_frames / frame_rate)).round('ms'),
            'duration_min': n_frames / frame_rate / 60,
            'fps': frame_rate,
            'n_frames': n_frames,
            'n_bins': int(value('sampleperchannel')),
            'min_range_m': float(value('minrange')),
            'max_range_m': float(value('maxrange')),
            'min_value': vmin,
            'max_value': vmax,
            'file': mat_file.name,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('deployment_path', type=Path)
    args = parser.parse_args()

    mat_files = find_aris_mat_files(args.deployment_path)
    if not mat_files:
        print(f'no ARIS .mat files found under {args.deployment_path}')
        return

    rows = []
    for mat_file in mat_files:
        print(f'reading {mat_file.name}')
        rows.append(file_stats(mat_file))
    df = pd.DataFrame(rows).sort_values('start')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', None)
    print()
    print(df.to_string(index=False, float_format=lambda x: f'{x:.1f}'))


if __name__ == '__main__':
    main()
