#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=5
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

# The hyperparameter settings reproduce the mean rank results
# reported in [Nickel, Kiela, 2017]
# For MAP results, replace the learning rate parameter with -lr 2.0

python3 embed.py \
       -dim 10 \
       -lr 2.0 \
       -epochs 800 \
       -negs 10 \
       -burnin 10 \
       -nproc "${NTHREADS}" \
       -distfn poincare \
       -dset wordnet/noun_closure.train.tsv \
       -dset_test wordnet/noun_closure.test.tsv \
       -fout model/nouns.10d.train \
       -batchsize 20 \
       -eval_each 10 \
