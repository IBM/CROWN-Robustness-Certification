#!/bin/bash

model=mnist
ntype=vanilla
neuron=20
for target in top2 least random; do
  for layer in 2 3; do
    for norm in 1 2 i; do
      for method in ours lip; do
        echo $model $ntype $layer $neuron $norm $method $target
      done
    done
  done
done

model=cifar
ntype=vanilla
for target in top2 least random; do
  for method in ours lip; do
    for norm in 1 2 i; do
      neuron=2048
      for layer in 5 6; do
        echo $model $ntype $layer $neuron $norm $method $target
      done
      neuron=1024
      layer=7
      echo $model $ntype $layer $neuron $norm $method $target
    done
  done
done

model=cifar
ntype=vanilla
neuron=1024
target=untargeted
layer=5
for norm in 1 2 i; do
  for method in ours lip; do
    echo $model $ntype $layer $neuron $norm $method $target
  done
done

model=mnist
ntype=vanilla
neuron=1024
target=untargeted
layer=3
for norm in 1 2 i; do
  if [ "$norm" == "2" ]; then
    for method in ours lip; do
      echo $model $ntype $layer $neuron $norm $method $target
    done
  else
    for method in ours lip; do
      echo $model $ntype $layer $neuron $norm $method $target
    done
  fi
done

model=mnist
ntype=vanilla
neuron=1024
for target in top2 least random; do
  for layer in 2 3 4; do
    for norm in 1 2 i; do
      if [ "$layer" == "4" ]; then
        for method in ours lip; do
          echo $model $ntype $layer $neuron $norm $method $target
        done
      else
        for method in ours lip; do
          echo $model $ntype $layer $neuron $norm $method $target
        done
      fi
    done
  done
done

