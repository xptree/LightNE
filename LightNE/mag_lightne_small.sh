#!/bin/bash

set -x

EXE=./LightNE
DATA=../data_bin
LOG=log_mag

if [ -z "$1" ]; then
    INPUT="$DATA/mag.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="mag.ne"
    PROOUT="mag.pro"
else
    NEOUT="$2.ne"
    PROOUT="$2.pro"
fi

if [ -z "$3" ]; then
  LABEL=$DATA/mag.label.npz
else
  LABEL=$3
fi

[ ! -f $INPUT ] && python ../util/x2adj.py --file $DATA/mag.edge --output $INPUT

mkdir -p $LOG
NOW=$(date +"%Y-%m-%d")

(/usr/bin/time -p numactl -i all $EXE -walksperedge 1 -walklen 10 -rounds 1 -s -m \
  -ne_out "" -pro_out $PROOUT -rank 256 -dim 128 -order 10 -sample_ratio 0.077 -mem_ratio 0.5 -negative 1 --sparse_project 0 \
  -ne_method netsmf -sample 1 -upper 0 -analyze 1 -tablesz 1717986918 -power_iteration 1 -oversampling 10 $INPUT) |& tee -a $LOG/$NOW.log


(python predict.py --label $LABEL --embedding $PROOUT --seed 0 --C 10 --start-train-ratio 0.001 --stop-train-ratio 0.001 --num-train-ratio 1 --num-split 2 --binary) |& tee -a $LOG/$NOW.log
(python predict.py --label $LABEL --embedding $PROOUT --seed 0 --C 10 --start-train-ratio 0.01 --stop-train-ratio 0.01 --num-train-ratio 1 --num-split 2 --binary) |& tee -a $LOG/$NOW.log
(python predict.py --label $LABEL --embedding $PROOUT --seed 0 --C 10 --start-train-ratio 0.1 --stop-train-ratio 1 --num-train-ratio 2 --num-split 2 --binary) |& tee -a $LOG/$NOW.log
