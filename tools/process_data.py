from pathlib import Path

from dragon_processing import Config, process_all

config = Config(
    raw=Path('/Volumes/LeConte Pro/dragon_data/raw/20260817_1800'),
    proc=Path('/Volumes/LeConte Pro/dragon_data/proc/20260817_1800_dragon'),
)

process_all(
    config,
    overwrite=False,
    basic_plots=False,
    include_aris=True,
)
