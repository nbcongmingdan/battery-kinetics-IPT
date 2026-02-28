Project Overview

This is a small video-tracking kinematics/energy analysis project. It includes several Jupyter Notebooks and three main scripts:
- track_battery.py: track a battery centroid and output speed/trajectory data
- track_rigidbody.py: track rigid-body position and orientation, compute kinetic/potential energy
- track_spin.py: track rotation angle and output angular-velocity CSV

Dependencies

Python 3.9+ is recommended.

1) (Optional) create and activate a virtual environment
2) Install dependencies
   pip install -r requirements.txt

Usage

1) track_spin.py (CLI arguments)
   python track_spin.py --input path/to/video.mp4 --csv angle_omega.csv --debug_video debug.mp4 --show

2) track_battery.py / track_rigidbody.py (run directly)
   These two scripts use VIDEO_PATH at the top of the file. Update it to a local video path before running:
   python track_battery.py
   python track_rigidbody.py

Outputs

- track_battery.py: outputs trajectory.txt
- track_rigidbody.py: outputs rigidbody_data.txt and total_energy.png
- track_spin.py: outputs angle_omega.csv (optional debug video)

Notebooks

This repo contains multiple .ipynb files for analysis and plotting. To run notebooks, install Jupyter (included in requirements.txt):
- pip install jupyter
