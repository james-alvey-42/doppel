#!/bin/bash
psigs=(0.01 0.05 0.32)
for index in ${!psigs[*]}
do
    echo "Running: psig | ${psigs[$index]}"
    python traditional_ts.py ${psigs[$index]}
done