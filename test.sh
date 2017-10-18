#!/bin/bash

for laM in 2 5
do 
	for laS in 4 9
	do
		echo 0.0000001 $((1.0/$((2.0*$laS)))) $((1.0/$laS)) $((2.0/$laS)) 10000
	done
done
