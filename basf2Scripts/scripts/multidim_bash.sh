#!/bin/bash

declare -x x=10
declare -x y=-0.6794
declare -x z=0.0
declare -x psi=296.2887
declare -x theta=328.9675
declare -x phi=84.0527

cd /home/ihaide/Masterarbeit/basf2Scripts

for i in {0..100}
do
	basf2 multidim_gauss_ogun.py -- --i=$i --var=x --x=$x --y=$y --z=$z --phi=$phi --theta=$theta --psi=$psi
done
