# Dragon Data Processing

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/PolarCoasts/dragon_processing.git
cd dragon_processing
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

To use the environment in Jupyter, register it as a kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name dragon-processing --display-name "Python (dragon_processing)"
```

## Processing Data

Set the `raw` and `proc` paths and the `process_all` options in `tools/process_data.py`, then run:

```bash
python tools/process_data.py
```

With `include_aris=True`, each `.aris` file must first be exported to `.mat` with MATLAB.

## Tools

| Tool | Purpose |
|---|---|
| `tools/aris_timeline.py` | Print statistics for each ARIS recording in a deployment |
| `tools/aris_filter_staging.py` | Render ARIS frames with intensity and range histograms, for choosing filter parameters |
| `tools/aris_sonargram.py` | Render final ARIS frames with the filtered peak spline |
| `tools/make_video.py` | Compile rendered frames into an `.mp4` (requires ffmpeg) |
| `tools/view_netcdf.ipynb` | Open a NetCDF file in a notebook |

```bash
python tools/aris_timeline.py <deployment_path>
python tools/aris_filter_staging.py <nc_file> <output_dir> [options]
python tools/aris_sonargram.py <nc_file> <output_dir> [options]
```

Add `--help` to the two render tools to list their options.

`tools/run_filter_staging.py` and `tools/run_sonargram.py` are templates with every option set to its default. Copy one to `run/` with a numbered name, such as `run/run_sonargram_01.py`, edit its values, then run it. `tools/make_video.py` is set up the same way.
