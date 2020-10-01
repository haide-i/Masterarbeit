#!/bin/bash

declare -x x=17.87906277
declare -x y=0.0
declare -x z=46.95203739
declare -x psi=291.6042586
declare -x theta=20.65293169
declare -x phi=0.0

source ~/.bashrc
conda activate test_basf2
source /cvmfs/belle.cern.ch/tools/b2setup release-05-00-01
cd /home/ihaide/Masterarbeit/basf2Scripts

for ((i=$1;i<=$2;i++)) #from 0 to 1800 for pos to get ranges x +- 5, y +- 0.9, z +- 5, from 0 to 100 for mom for theta +- 5, psi +- 5
do
	basf2 gridgeneratemom.py -- --i=$i --x=$x --y=$y --z=$z --phi=$phi --theta=$theta --psi=$psi
done
