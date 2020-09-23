#!/bin/bash

declare -x x=10
declare -x y=-0.6794
declare -x z=0.0
declare -x psi=296.2887
declare -x theta=328.9675
declare -x phi=84.0527

#source /cvmfs/belle.cern.ch/tools/b2setup release-05-00-01

cd /home/ihaide/Masterarbeit/basf2Scripts
 
for i in {8001..10000}
do
	python runOpticalGunGauss.py --i=$i --var=theta --x=$x --y=$y --z=$z --psi=$psi --theta=$theta --phi=$phi --numPhotons=50000
done
