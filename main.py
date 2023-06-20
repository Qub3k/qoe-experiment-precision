# GTP-4 was used for some part of the code correction/testing etc.

import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg
from scipy.optimize import curve_fit

import random
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jax.scipy.special import betaln



################### start of $\ell$ metric #############################
def one_or_nan(x):
    """
    Construct a "mask" array with the same dimension as x, with element NaN
    where x has NaN at the same location; and element 1 otherwise.
    :param x: array_like
    :return: an array with the same dimension as x
    """
    y = np.ones(x.shape)
    y[np.isnan(x)] = float('nan')
    return y


def get_sos_j(sig_r_j, o_ji):
    """
    Compute SOS (standard deviation of score) for PVS j
    :param sig_r_j:
    :param o_ji:
    :return: array containing the SOS for PVS j
    """
    den = np.nansum(one_or_nan(o_ji) /
                    np.tile(sig_r_j ** 2, (o_ji.shape[1], 1)).T, axis=1)
    s_j_std = 1.0 / np.sqrt(np.maximum(0., den))
    return s_j_std


def weighed_nanmean_2d(a, wts, axis):
    """
    Compute the weighted arithmetic mean along the specified axis, ignoring
    NaNs. It is similar to numpy's nanmean function, but with a weight.
    :param a: 1D array.
    :param wts: 1D array carrying the weights.
    :param axis: either 0 or 1, specifying the dimension along which the means
    are computed.
    :return: 1D array containing the mean values.
    copied from: https://github.com/Netflix/sureal/blob/36f73c0e20c538379c52b8681b68a97cc7667357/itut_p913_demo/demo.py
    """

    assert len(a.shape) == 2
    assert axis in [0, 1]
    d0, d1 = a.shape
    if axis == 0:
        return np.divide(
            np.nansum(np.multiply(a, np.tile(wts, (d1, 1)).T), axis=0),
            np.nansum(np.multiply(~np.isnan(a), np.tile(wts, (d1, 1)).T), axis=0)
        )
    elif axis == 1:
        return np.divide(
            np.nansum(np.multiply(a, np.tile(wts, (d0, 1))), axis=1),
            np.nansum(np.multiply(~np.isnan(a), np.tile(wts, (d0, 1))), axis=1),
        )
    else:
        assert False


def run_alternating_projection(o_ji, print = 1):
    """
    Run Alternating Projection (AP) algorithm.
    :param o_ji: 2D numpy array containing raw votes. The first dimension
    corresponds to the PVSs (j); the second dimension corresponds to the
    subjects (i). If a vote is missing, the element is NaN.
    :return: dictionary containing results keyed by 'mos_j', 'sos_j', 'bias_i'
    and 'inconsistency_i'.
    copied from: https://github.com/Netflix/sureal/blob/36f73c0e20c538379c52b8681b68a97cc7667357/itut_p913_demo/demo.py
    """
    J, I = o_ji.shape

    # video by video, estimate MOS by averaging over subjects
    psi_j = np.nanmean(o_ji, axis=1)  # mean marginalized over i

    # subject by subject, estimate subject bias by comparing with MOS
    b_ji = o_ji - np.tile(psi_j, (I, 1)).T
    b_i = np.nanmean(b_ji, axis=0)  # mean marginalized over j

    MAX_ITR = 1000
    DELTA_THR = 1e-8
    EPSILON = 1e-8

    itr = 0
    while True:

        psi_j_prev = psi_j

        # subject by subject, estimate subject inconsistency by averaging the
        # residue over stimuli
        r_ji = o_ji - np.tile(psi_j, (I, 1)).T - np.tile(b_i, (J, 1))
        sig_r_i = np.nanstd(r_ji, axis=0)
        sig_r_j = np.nanstd(r_ji, axis=1)

        # video by video, estimate MOS by averaging over subjects, inversely
        # weighted by residue variance
        w_i = 1.0 / (sig_r_i ** 2 + EPSILON)
        # mean marginalized over i:
        psi_j = weighed_nanmean_2d(o_ji - np.tile(b_i, (J, 1)), wts=w_i, axis=1)

        # subject by subject, estimate subject bias by comparing with MOS,
        # inversely weighted by residue variance
        b_ji = o_ji - np.tile(psi_j, (I, 1)).T
        # mean marginalized over j:
        b_i = np.nanmean(b_ji, axis=0)

        itr += 1

        delta_s_j = linalg.norm(psi_j_prev - psi_j)

        if print == 1:
            msg = 'Iteration {itr:4d}: change {delta_psi_j}, psi_j {psi_j}, ' \
                  'b_i {b_i}, sig_r_i {sig_r_i}'.format(
                itr=itr, delta_psi_j=delta_s_j, psi_j=np.mean(psi_j),
                b_i=np.mean(b_i), sig_r_i=np.mean(sig_r_i))

            sys.stdout.write(msg + '\r')
            sys.stdout.flush()
            sys.stdout.write("\n")

        if delta_s_j < DELTA_THR:
            break

        if itr >= MAX_ITR:
            break

    psi_j_std = get_sos_j(sig_r_j, o_ji)
    mean_b_i = np.mean(b_i)
    b_i -= mean_b_i
    psi_j += mean_b_i

    return {
        'mos_j': list(psi_j),
        'sos_j': list(psi_j_std),
        'bias_i': list(b_i),
        'inconsistency_i': list(sig_r_i),
        'raw_errors': np.array(r_ji),
    }


def precision_l(scores):
    """
    Run alternating projection and extracts statistics used to score precision
    :param scores: numpy matrix with rows as pvs and columns as subjects
    :return: dictionary with:
      l: precision measure
      n: number of subjects
      std: standard deviation of precision measure
    """
    model_par = run_alternating_projection(scores, 0)
    return{
        'l': np.mean(model_par['inconsistency_i']),
        'std': np.std(model_par['inconsistency_i']),
        'n': len(model_par['inconsistency_i'])
    }


def compar_by_l(data_frame, experiment_id, score, pvs, subject):
    """
    Compare two (exactly two) experiments
    :param data_frame: input data frame with long data structure. This DF has to contain at least four columns, experiment id, score, pvs, and subject
    :param experiment_id: name of the column with experiment ID
    :param score: name of the column with score
    :param pvs: name of the column with PVS ID
    :param subject: name of the column with subject ID
    :return: dictionary with:
      t_stat: is t-test statistics
      p_value: is t-test p-value
    """
    experiments = np.unique(data_frame[experiment_id])
    if len(experiments) != 2:
        return {
            't_stat': -1,
            'p_value': -1
        }
    else:
        df_pivot = data_frame[data_frame[experiment_id] == experiments[0]].pivot(index=pvs, columns=subject, values=score)
        matrix_1 = df_pivot.values
        df_pivot = data_frame[data_frame[experiment_id] == experiments[1]].pivot(index=pvs, columns=subject, values=score)
        matrix_2 = df_pivot.values
        prec_l_1 = precision_l(matrix_1)
        prec_l_2 = precision_l(matrix_2)
        t_stat, p_value = stats.ttest_ind_from_stats(prec_l_1['l'], prec_l_1['std'], prec_l_1['n'],
                                                     prec_l_2['l'], prec_l_2['std'], prec_l_2['n'],
                                                     equal_var=False)

        return {
            't_stat': t_stat,
            'p_value': p_value
        }

################# end of $\ell$ metric ################################

################ start metric $a$ ################################

def precision_a(scores):
    """
    Estimates the SOS parameter a and its standard error for a subjective experiment.

    Parameters
    ----------
    scores : n x k matrix of user ratings in one experiment with n subjects and k stimuli / test conditions
        An experiment is described by the matrix of user ratings (for n subjects and for k stimuli). Row i reflects ratings for stimuli i.

    Returns
    -------
    sosa : float
        Estimate of the SOS parameter a.
    stderr : float
        Standard error of the SOS parameter estimation.
    n : integer
        Number of test conditions (i.e. data points) of the estimation

    """

    def vos_fun(mos, a):
        return a * (5 - mos) * (mos - 1)

    mos, vos = scores.mean(axis=1), scores.var(axis=1)
    popt, pcov = curve_fit(vos_fun, mos, vos, bounds=(0, 1.0), absolute_sigma=True)
    stderr = np.sqrt(pcov[0][0])
    sosa = popt[0]
    n = len(mos)
    return {
        'a': sosa,
        'std': stderr,
        'n': n
    }

def compar_by_a(data_frame, experiment_id, score, pvs, subject):
    """
    Compare two (exactly two) experiments
    :param data_frame: input data frame with long data structure. This DF has to contain at least four columns, experiment id, score, pvs, and subject
    :param experiment_id: name of the column with experiment ID
    :param score: name of the column with score
    :param pvs: name of the column with PVS ID
    :param subject: name of the column with subject ID
    :return: dictionary with:
      t_stat: is t-test statistics
      p_value: is t-test p-value
    """
    experiments = np.unique(data_frame[experiment_id])
    if len(experiments) != 2:
        return {
            't_stat': -1,
            'p_value': -1
        }
    else:
        df_pivot = data_frame[data_frame[experiment_id] == experiments[0]].pivot(index=pvs, columns=subject, values=score)
        matrix_1 = df_pivot.values
        df_pivot = data_frame[data_frame[experiment_id] == experiments[1]].pivot(index=pvs, columns=subject, values=score)
        matrix_2 = df_pivot.values
        prec_l_1 = precision_a(matrix_1)
        prec_l_2 = precision_a(matrix_2)
        t_stat, p_value = stats.ttest_ind_from_stats(prec_l_1['a'], prec_l_1['std'], prec_l_1['n'],
                                                     prec_l_2['a'], prec_l_2['std'], prec_l_2['n'],
                                                     equal_var=False)

        return {
            't_stat': t_stat,
            'p_value': p_value
        }

################# end of $a$ metric ################################

################ start metric $g$ ################################

def logbinom(n, k):
    '''
    log of `n choose k`

    '''
    return -jnp.log1p(n) - betaln(n - k + 1., k + 1.)


def vmin(psi: jnp.ndarray) -> jnp.ndarray:
    return (jnp.ceil(psi) - psi) * (psi - jnp.floor(psi))


def vmax(psi: jnp.ndarray) -> jnp.ndarray:
    return (psi - 1.) * (5 - psi)


def _C(Vmax, Vmin):
    return 3. / 4. * Vmax / (Vmax - Vmin)


def _gsd_log_prob(psi, rho, k):
    index = jnp.arange(0, 6)

    Vmin = vmin(psi)
    Vmax = vmax(psi)
    C = _C(Vmax, Vmin)
    beta_bin_part = logbinom(4, k - 1.) + jnp.sum(
        jnp.where(
            index <= k - 2,
            jnp.log((psi - 1) * rho / 4 + index * (C - rho)),
            0.
        )
    ) + jnp.sum(
        jnp.where(
            index <= 4 - k,
            jnp.log((5. - psi) * rho / 4 + index * (C - rho)),
            0.
        )
    ) - jnp.sum(
        jnp.where(
            index <= 3,
            jnp.log(rho + index * (C - rho)),
            0.
        )
    )

    min_var_part = jax.nn.relu(1. - jnp.abs(k - psi))
    log_min_var_part = jnp.where(rho < C, 0., jnp.log(rho - C)) - \
                       jnp.log1p(-C) + jnp.log(min_var_part)
    log_bin_part = jnp.log1p(-rho) - jnp.log1p(-C) + logbinom(4, k - 1.) + \
                   (k - 1) * (jnp.log(psi - 1) - jnp.log(4)) + \
                   (5 - k) * (jnp.log(5 - psi) - jnp.log(4))

    almost_neg_inf = np.log(1e-10)
    logmix = jnp.logaddexp(jnp.where(min_var_part == 0, almost_neg_inf, log_min_var_part), log_bin_part)

    logmix = jnp.where(min_var_part == 0, log_bin_part, logmix)
    return jnp.where(rho < C, beta_bin_part, logmix)


class GSDParams(NamedTuple):
    psi: float
    rho: float


@jax.jit
def fit_moments(data: jnp.ndarray) -> GSDParams:
    psi = jnp.dot(data, jnp.arange(1, 6)) / jnp.sum(data)
    V = jnp.dot(data, jnp.arange(1, 6) ** 2) / jnp.sum(data) - psi ** 2
    return GSDParams(psi=psi, rho=(vmax(psi) - V) / (vmax(psi) - vmin(psi)))


v_gsd_log_prob = jax.vmap(_gsd_log_prob, (None, None, 0), (0))


class _OptState(NamedTuple):
    params: GSDParams
    prevoious_params: GSDParams
    count: int


@jax.jit
def fit_grad_bogdan(data: jnp.ndarray, max_iterations=100) -> GSDParams:
    def ll(theta: GSDParams) -> float:
        logits = v_gsd_log_prob(theta.psi, theta.rho, jnp.arange(1, 6))
        return jnp.dot(data, logits) / jnp.sum(data)

    grad_ll = jax.grad(ll)
    theta0 = fit_moments(data)
    rate = jnp.concatenate([jnp.zeros((1,)), jnp.logspace(-15, 5, 20, base=2.)])

    def update(tg, t, lo, hi):
        '''

        :param tg: gradient
        :param t: theta parametry
        :param lo: low bound
        :param hi: upper bound
        :return: updated params
        '''
        nt = t + rate * jnp.where(jnp.isnan(tg), 0., tg)
        mask = jnp.logical_and(nt > lo, nt < hi)
        return jnp.where(mask, nt, nt[0])

    # TODO handle special cases
    lo = GSDParams(psi=1., rho=0.)
    hi = GSDParams(psi=5., rho=1.)

    def body_fun(state: _OptState) -> _OptState:
        t, _, count = state
        g = grad_ll(t)
        new_theta = tree.tree_map(update, g, t, lo, hi)
        new_lls = jax.vmap(ll)(new_theta)
        max_idx = jnp.argmax(jnp.where(jnp.isnan(new_lls), -jnp.inf, new_lls))
        return _OptState(params=jax.tree_map(lambda t: t[max_idx], new_theta),
                         prevoious_params=t,
                         count=count + 1
                         )

    def cond_fun(state: _OptState) -> bool:
        tn, tnm1, c = state
        should_stop = jnp.logical_or(jnp.all(jnp.array(tn) == jnp.array(tnm1)), c > max_iterations)
        # stop on NaN
        should_stop = jnp.logical_or(should_stop, jnp.any(jnp.isnan(jnp.array(tn))))
        return jnp.logical_not(should_stop)

    opt_state = jax.lax.while_loop(cond_fun, body_fun,
                                   _OptState(params=theta0,
                                             prevoious_params=jax.tree_map(lambda _: jnp.inf, theta0),
                                             count=0
                                             )
                                   )
    return opt_state


def precision_g(scores):
    """
    Estimates the SOS parameter a and its standard error for a subjective experiment.

    Parameters
    ----------
    scores : n x k matrix of user ratings in one experiment with n subjects and k stimuli / test conditions
        An experiment is described by the matrix of user ratings (for n subjects and for k stimuli). Row i reflects ratings for stimuli i.

    Returns
    -------
    sosa : float
        Estimate of the SOS parameter a.
    stderr : float
        Standard error of the SOS parameter estimation.
    n : integer
        Number of test conditions (i.e. data points) of the estimation

    """
    n, _ = scores.shape
    rho = np.zeros(n)
    for i in range(n):
        scores_tmp = np.array([np.sum(scores[i,:] == 1), np.sum(scores[i,:] == 2), np.sum(scores[i,:] == 3),
                 np.sum(scores[i,:] == 4), np.sum(scores[i,:] == 5)])
        gsd_hat = fit_grad_bogdan(scores_tmp)
        rho[i] = gsd_hat.params[1]
    g = np.nanmean(rho)
    stdg = np.nanstd(rho)
    n = len(rho)
    return {
        'g': g,
        'std': stdg,
        'n': n
    }


def compar_by_g(data_frame, experiment_id, score, pvs, subject):
    """
    Compare two (exactly two) experiments
    :param data_frame: input data frame with long data structure. This DF has to contain at least four columns, experiment id, score, pvs, and subject
    :param experiment_id: name of the column with experiment ID
    :param score: name of the column with score
    :param pvs: name of the column with PVS ID
    :param subject: name of the column with subject ID
    :return: dictionary with:
      t_stat: is t-test statistics
      p_value: is t-test p-value
    """
    experiments = np.unique(data_frame[experiment_id])
    if len(experiments) != 2:
        return {
            't_stat': -1,
            'p_value': -1
        }
    else:
        df_pivot = data_frame[data_frame[experiment_id] == experiments[0]].pivot(index=pvs, columns=subject, values=score)
        matrix_1 = df_pivot.values
        df_pivot = data_frame[data_frame[experiment_id] == experiments[1]].pivot(index=pvs, columns=subject, values=score)
        matrix_2 = df_pivot.values
        prec_1 = precision_g(matrix_1)
        prec_2 = precision_g(matrix_2)
        t_stat, p_value = stats.ttest_ind_from_stats(prec_1['g'], prec_1['std'], prec_1['n'],
                                                     prec_2['g'], prec_2['std'], prec_2['n'],
                                                     equal_var=False)

        return {
            't_stat': t_stat,
            'p_value': p_value
        }

################# end of $g$ metric ################################

if __name__ == '__main__':
    # Example how to calculate l metric for a matrix of scores (rows are PVSs and columns are subjects)
    scores = np.ceil(5*np.random.uniform(0, 1, size = [10, 24]))
    a = precision_l(scores)
    print(a['l'])
    g = precision_g(scores)
    print(g)

    print(precision_a(scores))

    # Example how to use comparison metrix l for comparing two experiments
    # Generate data
    n = 200
    # Two different experiments
    exp = np.repeat([0, 1], int(n/2))
    # 20 subjects per experiment
    subject = np.tile(np.arange(1, 21), int(n/20))
    # 100 PVS per experiment
    pvs = np.repeat(np.arange(10), int(n/10))
    np.random.seed(0)  # For reproducibility
    # random score per PVS and subject
    score = np.random.choice(np.arange(1, 6), size=n)

    # Create DataFrame, we assume this DataFrame is read from files/databases/etc.
    df = pd.DataFrame({'exp': exp, 'pvs': pvs, 'subject': subject, 'score': score})

    # Pivot DataFrame, we have to extract data per experiment and convert it to wide format
    df_pivot = df[df['exp'] == 0].pivot(index='pvs', columns='subject', values='score')
    # Convert to numpy array, the precision measure are prepared for numpay array
    matrix = df_pivot.values

    # example of calculating 'l' measure for two experiments
    print(precision_l(matrix)['l'])
    df_pivot = df[df['exp'] == 1].pivot(index='pvs', columns='subject', values='score')
    matrix = df_pivot.values
    print(precision_l(matrix)['l'])
    print(compar_by_l(df, 'exp', 'score', 'pvs', 'subject'))
    print(compar_by_a(df, 'exp', 'score', 'pvs', 'subject'))
    print(compar_by_g(df, 'exp', 'score', 'pvs', 'subject'))
