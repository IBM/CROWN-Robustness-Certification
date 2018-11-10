##
## Copyright (C) IBM Corp, 2018
## Copyright (C) Huan Zhang <huan@huan-zhang.com>, 2018
## Copyright (C) Tsui-Wei Weng  <twweng@mit.edu>, 2018
## 
## This program is licenced under the Apache-2,0 licence,
## contained in the LICENCE file in this directory.
##

from numba import jit
import numpy as np
from general_fit import *

@jit(nopython=True)
def get_general_bounds(UBs, LBs, neuron_states, bounds_ul, ub_pn, lb_pn, ub_p, lb_p, ub_n, lb_n):
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

# cannot unify the bounds calculation functions due to limitations of numba
@jit(nopython=True)
def get_relu_bounds(UBs, LBs, neuron_states, bounds_ul):
    ub_pn = relu_ub_pn
    lb_pn = relu_lb_pn
    ub_p  = relu_ub_p
    lb_p  = relu_lb_p
    ub_n  = relu_ub_n
    lb_n  = relu_lb_n
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

@jit(nopython=True)
def get_tanh_bounds(UBs, LBs, neuron_states, bounds_ul):
    ub_pn = lambda u, l: general_ub_pn(u, l, act_tanh, act_tanh_d)
    lb_pn = lambda u, l: general_lb_pn(u, l, act_tanh, act_tanh_d)
    ub_p  = lambda u, l: general_ub_p(u, l, act_tanh, act_tanh_d)
    lb_p  = lambda u, l: general_lb_p(u, l, act_tanh, act_tanh_d)
    ub_n  = lambda u, l: general_ub_n(u, l, act_tanh, act_tanh_d)
    lb_n  = lambda u, l: general_lb_n(u, l, act_tanh, act_tanh_d)
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

@jit(nopython=True)
def get_sigmoid_bounds(UBs, LBs, neuron_states, bounds_ul):
    ub_pn = lambda u, l: general_ub_pn(u, l, act_sigmoid, act_sigmoid_d)
    lb_pn = lambda u, l: general_lb_pn(u, l, act_sigmoid, act_sigmoid_d)
    ub_p  = lambda u, l: general_ub_p(u, l, act_sigmoid, act_sigmoid_d)
    lb_p  = lambda u, l: general_lb_p(u, l, act_sigmoid, act_sigmoid_d)
    ub_n  = lambda u, l: general_ub_n(u, l, act_sigmoid, act_sigmoid_d)
    lb_n  = lambda u, l: general_lb_n(u, l, act_sigmoid, act_sigmoid_d)
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

@jit(nopython=True)
def get_arctan_bounds(UBs, LBs, neuron_states, bounds_ul):
    ub_pn = lambda u, l: general_ub_pn(u, l, act_arctan, act_arctan_d)
    lb_pn = lambda u, l: general_lb_pn(u, l, act_arctan, act_arctan_d)
    ub_p  = lambda u, l: general_ub_p(u, l, act_arctan, act_arctan_d)
    lb_p  = lambda u, l: general_lb_p(u, l, act_arctan, act_arctan_d)
    ub_n  = lambda u, l: general_ub_n(u, l, act_arctan, act_arctan_d)
    lb_n  = lambda u, l: general_lb_n(u, l, act_arctan, act_arctan_d)
    # step 1: get indices of three classes of neurons
    upper_k = bounds_ul[0]
    upper_b = bounds_ul[1]
    lower_k = bounds_ul[2]
    lower_b = bounds_ul[3]
    idx_p = np.nonzero(neuron_states == 1)[0]
    idx_n = np.nonzero(neuron_states == -1)[0]
    idx_pn = np.nonzero(neuron_states == 0)[0]
    upper_k[idx_pn], upper_b[idx_pn] = ub_pn(UBs[idx_pn], LBs[idx_pn])
    lower_k[idx_pn], lower_b[idx_pn] = lb_pn(UBs[idx_pn], LBs[idx_pn])
    upper_k[idx_p],  upper_b[idx_p]  = ub_p(UBs[idx_p], LBs[idx_p])
    lower_k[idx_p],  lower_b[idx_p]  = lb_p(UBs[idx_p], LBs[idx_p])
    upper_k[idx_n],  upper_b[idx_n]  = ub_n(UBs[idx_n], LBs[idx_n])
    lower_k[idx_n],  lower_b[idx_n]  = lb_n(UBs[idx_n], LBs[idx_n])
    return upper_k, upper_b, lower_k, lower_b

def init_layer_bound_relax_matrix_huan_general(Ws):
    nlayer = len(Ws)
    # preallocate all upper and lower bound slopes and intercepts
    bounds_ul = [None] * nlayer
    # first k is identity
    bounds_ul[0] = np.ones((4,Ws[0].shape[1]), dtype=np.float32)
    for i in range(1,nlayer):
        bounds_ul[i] = np.empty((4,Ws[i].shape[1]), dtype=np.float32)
    return bounds_ul

# adaptive matrix version of get_layer_bound_relax
@jit(nopython=True)
def get_layer_bound_relax_adaptive_matrix_huan_general_optimized(Ws,bs,UBs,LBs,neuron_state,nlayer,bounds_ul,x0,eps,q_np,get_bounds):
    assert nlayer >= 2
    assert nlayer == len(Ws) == len(bs) == len(UBs) == len(LBs) == (len(neuron_state) + 1) == len(bounds_ul)
    assert q_np <= 2 or q_np == np.inf

    # step 2: compute slopes and intercepts for upper and lower bounds
    # only need to create upper/lower bounds' slope and intercept for this layer,
    # slopes and intercepts for previous layers have been stored
    # index: 0->slope for ub, 1->intercept for ub, 
    #        2->slope for lb, 3->intercept for lb
    get_bounds(UBs[nlayer-1], LBs[nlayer-1], neuron_state[nlayer - 2], bounds_ul[nlayer-1])

    # step 3: update matrix A (merged into one loop)
    # step 4: adding all constants (merged into one loop)
    constants_ub = np.copy(bs[-1]) # the last bias
    constants_lb = np.copy(bs[-1]) # the last bias

    # step 5: bounding l_n term for each layer
    UB_final = np.zeros_like(constants_ub)
    LB_final = np.zeros_like(constants_lb)
    # first A is W_{nlayer} D_{nlayer}
    # A_UB = Ws[nlayer-1] * diags[nlayer-1]
    A_UB = np.copy(Ws[nlayer-1])
    # A_LB = Ws[nlayer-1] * diags[nlayer-1]
    A_LB = np.copy(Ws[nlayer-1])
    for i in range(nlayer-1, 0, -1):
        # create intercepts array for this layer
        l_ub = np.empty_like(LBs[i])
        l_lb = np.empty_like(LBs[i])
        diags_ub = np.empty_like(bounds_ul[i][0,:])
        diags_lb = np.empty_like(bounds_ul[i][0,:])
        upper_k = bounds_ul[i][0]
        upper_b = bounds_ul[i][1]
        lower_k = bounds_ul[i][2]
        lower_b = bounds_ul[i][3]
        """
        if not np.isfinite(upper_k).all():
            print("upper_k nan detected", i)
            return UB_final, LB_final
        if not np.isfinite(upper_b).all():
            print("upper_b nan detected", i)
            return UB_final, LB_final
        if not np.isfinite(lower_k).all():
            print("lower_k nan detected", i)
            return UB_final, LB_final
        if not np.isfinite(lower_b).all():
            print("lower_b nan detected", i)
            print(lower_b)
            loc = np.argwhere(np.isinf(lower_b))
            print(loc)
            u = UBs[i][loc]
            l = LBs[i][loc]
            print(u, l)
            print(general_lb_p(u, l, act_tanh, act_tanh_d))
            print(general_ub_p(u, l, act_tanh, act_tanh_d))
            print(lower_b[loc])
            return UB_final, LB_final
        """
        # bound the term A[i] * l_[i], for each element
        for j in range(A_UB.shape[0]):
            # index for positive entries in A for upper bound
            idx_pos_ub = np.nonzero(A_UB[j] > 0)[0]
            # index for negative entries in A for upper bound
            idx_neg_ub = np.nonzero(A_UB[j] <= 0)[0]
            # index for positive entries in A for lower bound
            idx_pos_lb = np.nonzero(A_LB[j] > 0)[0]
            # index for negative entries in A for lower bound
            idx_neg_lb = np.nonzero(A_LB[j] <= 0)[0]
            # for upper bound, set the neurons with positive entries in A to upper bound
            diags_ub[idx_pos_ub]  = upper_k[idx_pos_ub]
            l_ub[idx_pos_ub] = upper_b[idx_pos_ub]
            # for upper bound, set the neurons with negative entries in A to lower bound
            diags_ub[idx_neg_ub] = lower_k[idx_neg_ub]
            l_ub[idx_neg_ub] = lower_b[idx_neg_ub]
            # for lower bound, set the neurons with negative entries in A to upper bound
            diags_lb[idx_neg_lb] = upper_k[idx_neg_lb]
            l_lb[idx_neg_lb] = upper_b[idx_neg_lb]
            # for lower bound, set the neurons with positve entries in A to lower bound
            diags_lb[idx_pos_lb] = lower_k[idx_pos_lb]
            l_lb[idx_pos_lb] = lower_b[idx_pos_lb]
            """
            if not np.isfinite(A_UB[j]).all():
                print("A_UB[j] nan detected", i, j)
                return UB_final, LB_final
            if not np.isfinite(A_LB[j]).all():
                print("A_LB[j] nan detected", i, j)
                return UB_final, LB_final
            if not np.isfinite(l_ub).all():
                print("l_ub nan detected", i, j)
                return UB_final, LB_final
            if not np.isfinite(l_lb).all():
                print("l_lb nan detected", i, j)
                return UB_final, LB_final
            """
            # compute the relavent terms
            UB_final[j] += np.dot(A_UB[j], l_ub)
            LB_final[j] += np.dot(A_LB[j], l_lb)
            # update the j-th row of A with diagonal matrice
            A_UB[j] = A_UB[j] * diags_ub
            # update A with diagonal matrice
            A_LB[j] = A_LB[j] * diags_lb
            """
            if not np.isfinite(UB_final).all():
                print("UB_final nan detected", i, j)
                return UB_final, LB_final
            if not np.isfinite(LB_final).all():
                print("LB_final nan detected", i, j)
                return UB_final, LB_final
            """
        # constants of previous layers
        constants_ub += np.dot(A_UB, bs[i-1])
        constants_lb += np.dot(A_LB, bs[i-1])
        """
        if not np.isfinite(constants_ub).all():
            print("constants_ub nan detected", i, j)
            return UB_final, LB_final
        if not np.isfinite(constants_lb).all():
            print("constants_lb nan detected", i, j)
            return UB_final, LB_final
        """
        # compute A for next loop
        # diags matrices is multiplied above
        A_UB = np.dot(A_UB, Ws[i-1])
        A_LB = np.dot(A_LB, Ws[i-1])
    # after the loop is done we get A0
    UB_final += constants_ub
    LB_final += constants_lb

    # step 6: bounding A0 * x
    x_UB = np.empty_like(UBs[0])
    x_LB = np.empty_like(LBs[0])
    
    Ax0_UB = np.dot(A_UB,x0)
    Ax0_LB = np.dot(A_LB,x0)
    for j in range(A_UB.shape[0]):        
        dualnorm_Aj_ub = np.linalg.norm(A_UB[j], q_np)
        dualnorm_Aj_lb = np.linalg.norm(A_LB[j], q_np)
        UB_final[j] += (Ax0_UB[j]+eps*dualnorm_Aj_ub)
        LB_final[j] += (Ax0_LB[j]-eps*dualnorm_Aj_lb)
    
    return UB_final, LB_final
