#!/bin/bash

set -x

EXE=./LightNE
DATA=../data_bin

if [ -z "$1" ]; then
    INPUT="$DATA/blogcatalog.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="blog.ne"
    PROOUT="blog.pro"
else
    NEOUT="$2.ne"
    PROOUT="$2.pro"
fi

if [ -z "$3" ]; then
  LABEL=$DATA/blogcatalog.mat
else
  LABEL=$3
fi

[ ! -f $INPUT ] && python ../util/x2adj.py --file $LABEL --output $INPUT

(/usr/bin/time -p numactl -i all $EXE -walksperedge 10000 -walklen 10 -rounds 1 -s -m \
  -ne_out $NEOUT -pro_out $PROOUT -ne_method netsmf -rank 4096 -dim 128 -order 10 \
  -analyze 1 -sample 1 -sample_ratio 2000 -upper 0 -tablesz 6679660000 $INPUT ) |& tee -a blog_lightne.log

python predict.py --label $LABEL --embedding $PROOUT --seed 0 --C 10 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --binary --dim 128
