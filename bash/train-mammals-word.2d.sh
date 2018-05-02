#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=1
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

# The hyperparameter settings reproduce the mean rank results 
# reported in [Nickel, Kiela, 2017]
# For MAP results, replace the learning rate parameter with -lr 2.0

python3 embed.py \
       -dim 2 \
       -lr 2.0 \
       -epochs 300 \
       -negs 50 \
       -burnin 10 \
       -nproc "${NTHREADS}" \
       -distfn poincare \
       -dset wordnet/mammal_closure.tsv \
       -fout model/mammals.2d \
       -batchsize 10 \
       -eval_each 10 \
       -word
