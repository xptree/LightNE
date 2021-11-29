#!/bin/bash

set -x

EXE=./LightNE
DATA=../data_bin

if [ -z "$1" ]; then
    INPUT="$DATA/youtube.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="youtube.ne"
    PROOUT="youtube.pro"
else
    NEOUT="$2.ne"
    PROOUT="$2.pro"
fi

if [ -z "$3" ]; then
  LABEL=$DATA/youtube.mat
else
  LABEL=$3
fi

[ ! -f $INPUT ] && python ../util/x2adj.py --file $LABEL --output $INPUT

(/usr/bin/time -v numactl -i all $EXE -walksperedge 200 -walklen 6 -rounds 1 -s -m \
  -ne_method netsmf -ne_out $NEOUT -pro_out $PROOUT -rank 512 -dim 128 -order 10 -negative 1 \
  -sample 1 -sample_ratio 1 -sparse_project 0 -upper 0 -analyze 1 -tablesz 1196177200 -power_iteration 1 -oversampling 50 $INPUT) |& tee -a youtube_lightne.log


python predict.py --label $LABEL --embedding $PROOUT --seed 0 --C 2 --start-train-ratio 1 --stop-train-ratio 10 --num-train-ratio 10 --dim 128  --num-split 5 --binary
