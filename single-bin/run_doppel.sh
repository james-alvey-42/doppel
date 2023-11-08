#!/bin/bash
backgrounds=(1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
amplitudes=( 4 4 4 4 6  6  6  6  8  8  8  8 10 10 10 10 12 12 12 12 14 14 14 14 14 14 14 14 14  14)
for index in ${!backgrounds[*]}
do
    echo "Running: bkg | ${backgrounds[$index]}, amp | ${amplitudes[$index]}"
    python doppel.py ${backgrounds[$index]} ${amplitudes[$index]}
    rm -r single-bin-store-${backgrounds[$index]}-${amplitudes[$index]}*
done