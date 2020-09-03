#!/bin/bash

file='startdata.txt'
while IFS= read -r x y z phi theta psi
do
	echo '$z'
done
