#!/bin/bash

declare -x x=10
declare -x y=-0.6794
declare -x z=0.0
declare -x psi=296.2887
declare -x theta=296.2887
declare -x phi=84.0527

for i in {0..100}
do
	basf2 noclass_ogun_gauss.py -- --i=$i --var=x --x=$x --y=$y --phi=$phi --theta=$theta --psi=$psi
done
