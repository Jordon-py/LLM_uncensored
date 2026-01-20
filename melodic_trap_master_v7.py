"""
Stereo Field Designer v3
------------------------
Entry point for the modular stereo imaging suite.

Usage:
  python melodic_trap_master_v7.py            # Launch GUI
  python melodic_trap_master_v7.py --cli --in "input.wav" --out "output.wav" --preset "Wide Dream"
"""

from stereo_field_designer_v3.stereo_field_designer import main


if __name__ == "__main__":
    main()
