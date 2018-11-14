CROWN: A Neural Network Verification Framework
--------------------
We proposed an new framework, **CROWN**, to **certify** robustness of neural networks with **general activation functions** including but not limited to ReLU, tanh, sigmoid, arctan, etc. **CROWN** is efficient and can deliver lower bounds of minimum adversarial distortions with guarantees (the so-called **certified lower bound** or **certified robustness**).

We compare **CROWN** with various certified lower bounds methods including [Global Lipschitz constant](https://arxiv.org/pdf/1312.6199.pdf) and [Fast-Lin](https://github.com/huanzhang12/CertifiedReLURobustness) and show that **CROWN** can certify much large lower bound than the Global Lipschitz constant based approach while improve the quality (up to 28%) of robustness lower bound on ReLU networks of state-of-the-art robustness certification algorithms Fast-Lin. We also compare **CROWN** with robustness score estimate [CLEVER](https://github.com/huanzhang12/CLEVER) and adversarial attack methods ([CW](https://github.com/carlini/nn_robust_attacks),[EAD](https://github.com/ysharma1126/EAD_Attack)). Please See Section 4 and Appendix E of our paper for more details.  

Cite our work:

Huan Zhang\*, Tsui-Wei Weng\*, Pin-Yu Chen, Cho-Jui Hsieh and Luca Daniel, "[**Efficient Neural Network Robustness Certification with General Activation Functions**](http://arxiv.org/abs/1811.00866)", NIPS 2018. (\* Equal Contribution)

```
@inproceedings{zhang2018crown,
  author = "Huan Zhang AND Tsui-Wei Weng AND Pin-Yu Chen AND Cho-Jui Hsieh AND Luca Daniel",
  title = "Efficient Neural Network Robustness Certification with General Activation Functions",
  booktitle = "Advances in Neural Information Processing Systems (NIPS)",
  year = "2018",
  month = "dec"
}
```

Prerequisites
-----------------------

The code is tested with python3 and TensorFlow v1.8 and v1.10. We suggest to
use Conda to manage your Python environments.  The following Conda packages are
required:

```
conda install pillow numpy scipy pandas tensorflow-gpu h5py
conda install --channel numba llvmlite numba
grep 'AMD' /proc/cpuinfo >/dev/null && conda install nomkl
```

You will also need to install Gurobi and its python bindings if you want to try the LP based methods. 

After installing prerequisites, clone our repository:

```
git clone https://github.com/huanzhang12/CROWN-Robustness-Certification.git
cd CROWN-Robustness-Certification
```

Our pretrained models can be download here:

```
wget http://jaina.cs.ucdavis.edu/datasets/adv/relu_verification/models_crown.tar
tar xvf models_crown.tar
```

This will create a `models` folder. We include all models reported in our paper.

How to Run
--------------------

We have provided an interfacing script, `run.sh` to reproduce the experiments in our paper.

```
Usage: ./run.sh model modeltype layer neuron norm solver target --activation ACT
```

* model: mnist or cifar
* modeltype: vanilla (undefended), adv\_retrain (adversarially trained)
* layer: number of layers (2,3,4 for MNIST and 5,6,7 for CIFAR)
* neuron: number of neurons for each layer (20 or 1024 for MNIST, 2048 for CIFAR)
* norm: p-norm, 1,2 or i (infinity norm)
* solver: adaptive (CROWN-adaptive for improved bounds for ReLU Networks), general (CROWN-general for networks with general activation functions)
* target: least, top2 (runner up), random, untargeted
* --activation: followed by a activation function of network (relu, tanh, sigmoid or arctan)

The main interfacing code is `main.py`, which provides additional options. Use `python main.py -h` to explore these options.

Examples
----------------

For example, to evaluate the Linf robustness of MNIST 3\*[1024] adversarially trained model using CROWN-adaptive on least likely targets, run

```
./run.sh mnist adv_retrain 3 1024 i adaptive least
```

A log file will be created in the `logs` folder. The last line of the log (starting with [L0]) will report the average
robustness lower bounds on 100 MNIST test images. Lines starting with [L1] reports per-image information.

```
 tail logs/mnist/3/mnist_adv_retrain_3_1024_Li_adaptive_least_none_*.log
```

```
[L0] model = models/mnist_3layer_relu_1024_adv_retrain, avg robustness_gx = 0.21916, numimage = 96, total_time = 85.4255
```

The adversarially trained model (with adversarial examples crafted by PGD with eps = 0.3) has a robustness lower bound of 0.21916.

Similarly, to evaluate the L1 robustness of MNIST 3\*[20] model with tanh activation on random targets using CROWN-general, run the following command:

```
./run.sh mnist vanilla 3 20 1 general random --activation tanh
```

The following result in log file is obtained:

```
[L0] model = models/mnist_3layer_tanh_20, avg robustness_gx = 1.42974, numimage = 97, total_time = 14.1968
```


Other notes
-------------------

Note that in our experiments we set the number of threads to 1 for a fair comparison to other methods.
To enable multithreaded computing, changing the number `1` in `run.sh` to the number of cores in your system.

```
NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```
The code in this repository is released under Apache-2.0 License and is intended to reproduce paper results only (see LICENSE for more details). An alternative version of CROWN can be found [here](https://github.com/huanzhang12/RecurJac-Jacobian-Bounds).

