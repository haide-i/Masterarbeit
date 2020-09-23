#!/bin/bash

declare -x x=-6.76046717
declare -x y=0.0
declare -x z=-14.31371242
declare -x psi=95.32992238
declare -x theta=281.79759198
declare -x phi=0.0

cd /home/ihaide/Masterarbeit/basf2Scripts

for i in {1601..1800}
do
	basf2 gridgeneratepos.py -- --i=$i --var=x --x=$x --y=$y --z=$z --phi=$phi --theta=$theta --psi=$psi
done
