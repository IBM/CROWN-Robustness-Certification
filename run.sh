#!/bin/bash

if [ "$#" -le 5 ]; then
  echo "Usage: $0 model modeltype layer neuron norm solver target"
  exit 1
fi

model=$1
modeltype=$2
layer=$3
neuron=$4
norm=$5
solver=$6
target=$7
additional=${@:8}
add_name=${additional// /_}
add_name=${add_name//-/}
if [[ -z $add_name ]]; then
  add_name="none"
fi
if [ "$solver" == "ours" ]; then
  opt=""
elif [ "$solver" == "lp" ]; then
  opt="--LP"
elif [ "$solver" == "lip" ]; then
  opt="--lipsbnd fast"
elif [ "$solver" == "spectral" ]; then
  opt="--method spectral"
elif [ "$solver" == "adaptive" ]; then
  opt="--method adaptive"
elif [ "$solver" == "general" ]; then
  opt="--method general"
else
  echo "Wrong solver $solver"
  exit 1
fi
if [ "$solver" == "lip" ]; then
# largest eps
  eps=0.05
  if [ "$model" == "cifar" ]; then
    eps=0.01
  fi
else
# initial test eps
  eps=0.01
  if [ "$model" == "cifar" ]; then
    eps=0.002
  fi
fi
if [ "$norm" == "2" ]; then
  eps=$(echo $eps*20 | bc)
  if [ "$model" == "cifar" ]; then
    eps=$(echo $eps*5 | bc)
  fi
fi
if [ "$norm" == "1" ]; then
  eps=$(echo $eps*30 | bc)
  if [ "$model" == "cifar" ]; then
    eps=$(echo $eps*5 | bc)
  fi
fi
output="${model}_${modeltype}_${layer}_${neuron}_L${norm}_${solver}_${target}_${add_name}_$(date +%m%d_%H%M%S)"
dir="logs/$model/$layer"
mkdir -p $dir
logfile=$dir/$output.log
echo $logfile

CMD="python3 main.py --numimage 100 --numlayer $layer --model $model --modeltype $modeltype --hidden $neuron --eps $eps --norm $norm $opt --targettype $target --warmup $additional"
echo $CMD
# NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 $CMD 2>&1 | tee $logfile
# NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $CMD 2>&1 | tee $logfile
NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $CMD >$logfile 2>$logfile.err
echo "Done $logfile"

