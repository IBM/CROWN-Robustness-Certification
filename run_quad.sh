#!/bin/bash

model=mnist
ntype=vanilla
neuron=20
for target in top2 least random; do
  for layer in 2; do
    for norm in 1 2 i; do
      for method in adaptive; do
        echo $model $ntype $layer $neuron $norm $method $target --quad
      done
    done
  done
done

model=mnist
ntype=vanilla
neuron=1024
for target in top2 least random; do
  for layer in 2; do
    for norm in 1 2 i; do
      for method in adaptive; do
        echo $model $ntype $layer $neuron $norm $method $target --quad
      done
    done
  done
done

