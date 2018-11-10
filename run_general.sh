#!/bin/bash

model=mnist
ntype=vanilla
neuron=20
for target in top2 least random; do
  for layer in 2 3; do
    for norm in 1 2 i; do
      for method in general; do
        for activation in tanh sigmoid arctan; do
          echo $model $ntype $layer $neuron $norm $method $target --activation $activation
        done
      done
    done
  done
done

model=cifar
ntype=vanilla
for target in top2 least random; do
  for method in general; do
    for norm in 1 2 i; do
      neuron=2048
      for layer in 5 6; do
        for activation in tanh arctan; do
          echo $model $ntype $layer $neuron $norm $method $target --activation $activation
        done
      done
      neuron=1024
      layer=7
      for activation in tanh arctan; do
        echo $model $ntype $layer $neuron $norm $method $target --activation $activation
      done
    done
  done
done

model=cifar
ntype=vanilla
neuron=1024
target=untargeted
layer=5
for norm in 1 2 i; do
  for method in general; do
    for activation in tanh arctan; do
      echo $model $ntype $layer $neuron $norm $method $target --activation $activation
    done
  done
done

model=mnist
ntype=vanilla
neuron=1024
target=untargeted
layer=3
for norm in 1 2 i; do
  for method in general; do
    for activation in tanh sigmoid arctan; do
      echo $model $ntype $layer $neuron $norm $method $target --activation $activation
    done
  done
done

model=mnist
ntype=vanilla
neuron=1024
for target in top2 least random; do
  for layer in 2 3 4; do
    for norm in 1 2 i; do
      for method in general; do
        for activation in tanh sigmoid arctan; do
          echo $model $ntype $layer $neuron $norm $method $target --activation $activation
        done
      done
    done
  done
done

