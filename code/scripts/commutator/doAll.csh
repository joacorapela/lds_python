#!/bin/csh

python doSmoothMouseTrajectory.py --bodypart snout
python doSmoothMouseTrajectory.py --bodypart tailbase
python doPlotSmoothedAngles.py --src_bodypart tailbase --dst_bodypart snout
