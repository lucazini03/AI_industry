from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import logging
import os
import time
import json
from scipy import stats
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import heapq

from ortools.linear_solver import pywraplp

# torch
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# lightning
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

# scikit-learn
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# General utility methods
# ==============================================================================

class LetMeTrainQuietlyWTF(logging.Filter):
    def filter(self, record):
        ok = "💡 Tip" not in record.getMessage()
        ok = ok and 'GPU available' not in record.getMessage()
        ok = ok and 'TPU available' not in record.getMessage()
        ok = ok and 'Trainer.fit' not in record.getMessage()
        return ok

logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(LetMeTrainQuietlyWTF())

def normal_sample_(mean, std, size=1, seed=42):
    np.random.seed(seed)
    return norm.rvs(size=size, loc=mean, scale=std)

# ==============================================================================
# Toy optimization problem
# ==============================================================================

def eval_predictions_(x, w, model=1):
    assert(model in (1, 2, 3))
    if model == 1:
        c0 = w**2 * x
        c1 = 0.5 * w * x**0
        # c1 = 0.5 * np.abs(w) * x**0
    elif model == 2:
        c0 = w * x
        c1 = 1 - c0
        # c0 = w * x
        # c1 = 0.5 * x**0
    elif model == 3:
        c0 = w * x
        c1 = 0.1 - c0
    res = np.stack((c0, c1), axis=-1)
    return res


def eval_ground_truth_(x):
    p0=[0., 0., 2.5]
    p1=[0.3, 0.8, 0]
    f0 = np.polynomial.Polynomial(p0)
    f1 = np.polynomial.Polynomial(p1)
    res = np.stack((f0(x), f1(x)), axis=-1)
    return res


def find_optimal_w_(x, model=1):
    y = eval_ground_truth_(x)
    wvals = np.linspace(0, 2, 1000)
    mse = []
    for w in wvals:
        yp = eval_predictions_(x, w, model=model)
        mse.append(np.mean(np.square(yp - y)))
    return wvals[np.argmin(mse)]


def get_decisions_(c):
    # Compute the optimal decisions
    x_0 = c[:, :, 0] <= c[:, :, 1]
    x = np.stack((x_0, 1 - x_0), axis=-1)
    return x


def draw_ground_truth_(x):
    y = eval_ground_truth_(x)
    plt.plot(x, y[:, 0], label=r'$y_0$', color='0.5', linestyle='-')
    plt.plot(x, y[:, 1], label=r'$y_1$', color='0.5', linestyle=':')


def draw_predictions_(x, w, model=1):
    y = eval_predictions_(x, w, model=model)
    plt.plot(x, y[:, 0], label=r'$\hat{y}_0$', color='tab:orange', linestyle='-')
    plt.plot(x, y[:, 1], label=r'$\hat{y}_1$', color='tab:orange', linestyle=':')


def draw(xmin=0, xmax=1, npoints=100, w=None, model=1, figsize=(10, 5)):
    x = np.linspace(xmin, xmax, npoints)
    if w is None:
        w = find_optimal_w_(x, model=model)
        print(f'Optimized theta: {w:.3f}')
    plt.figure(figsize=figsize)
    draw_ground_truth_(x)
    draw_predictions_(x, w)
    plt.grid(':')
    plt.legend()
    plt.show()



def plot_df_cols(data, scatter=False, figsize=None, legend=True, title=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    x = data.index
    plt.xlabel(data.index.name)
    # Plot all columns
    for cname in data.columns:
        y = data[cname]
        plt.plot(x, y, label=cname,
                 linestyle='-' if not scatter else '',
                 marker=None if not scatter else '.',
                 alpha=1 if not scatter else 0.3)
    # Add legend
    if legend and len(data.columns) <= 10:
        plt.legend(loc='best')
    plt.grid(':')
    # Add a title
    plt.title(title)
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


# ==============================================================================
# Drawing methods
# ==============================================================================

def plot_solcount(counts, label=None, figsize=None,
                  counts2=None, label2=None, print_total=False):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    if label and not label2:
        plt.xlabel(label)
    # Sort the counts by decreasing order
    counts = sorted(counts, reverse=True)
    # Histogram
    plt.step(np.arange(len(counts)), counts, where='post', label=label)
    if counts2 is not None:
        counts2 = sorted(counts2, reverse=True)
        plt.step(np.arange(len(counts2)), counts2, where='post', label=label2)
    # Make it compact
    plt.tight_layout()
    # Legend
    plt.legend()
    # Show
    plt.show()
    # Print total, if requested
    if print_total:
        s = f'Total: {np.sum(counts)}'
        if label is not None:
            s += f' ({label})'
        if counts2 is not None:
            s += f', {np.sum(counts2)}'
            if label2 is not None:
                s += f' ({label2})'
        print(s)
    return fig

def plot_histogram(data, label=None, bins=20, figsize=None,
                   data2=None, label2=None, print_mean=False,
                   override_x_label=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    if label and not label2:
        plt.xlabel(label)
    # Define bins
    rmin, rmax = data.min(), data.max()
    if data2 is not None:
        rmin = min(rmin, data2.min())
        rmax = max(rmax, data2.max())
    bins = np.linspace(rmin, rmax, bins)
    # Histogram
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    plt.step(edges[:-1], hist, where='post', label=label)
    if data2 is not None:
        hist2, edges2 = np.histogram(data2, bins=bins)
        hist2 = hist2 / np.sum(hist2)
        plt.step(edges2[:-1], hist2, where='post', label=label2)
    # Make it compact
    plt.tight_layout()
    # Legend
    plt.legend()
    # Show
    plt.show()
    # Print mean, if requested
    if print_mean:
        s = f'Mean: {np.mean(data):.3f}'
        if label is not None:
            s += f' ({label})'
        if data2 is not None:
            s += f', {np.mean(data2):.3f}'
            if label2 is not None:
                s += f' ({label2})'
        print(s)
    return fig


def draw_loss_landscape(losses,
                        w_min=-0.25,
                        w_max=1.5,
                        w_nvals=1000,
                        x_mean=0.54,
                        x_std=0.2,
                        batch_size=1,
                        model=1,
                        seed=42,
                        figsize=(14, 4)):
    # Draw samples
    x = normal_sample_(mean=x_mean, std=x_std, size=batch_size, seed=seed)
    x = x.reshape(1, -1)
    # Define w values
    w_vals = np.linspace(w_min, w_max, w_nvals)
    # Obtain true costs
    c_star = eval_ground_truth_(x)
    # Obtain the true optimal decisions
    y_star = get_decisions_(c_star)
    # Consider all ws
    lvals_per_w = []
    for w in w_vals:
        # Obtain estimated costs
        c_hat = eval_predictions_(x, w, model=model)
        # Prepare a data structure with all the loss values
        loss_values = [l(c_star, c_hat, y_star) for l in losses]
        # Store the loss vector
        lvals_per_w.append(loss_values)
    # Convert the loss data structure into an array
    lvals_per_w = np.array(lvals_per_w)
    # Draw a figure
    plt.figure(figsize=figsize)
    for i, l in enumerate(losses):
        plt.plot(w_vals, lvals_per_w[:, i], label=str(l))
    plt.title(f'number of examples: {batch_size}')
    plt.xlabel(r'$\theta$')
    plt.legend()
    plt.grid(':')
    plt.show()



def plot_training_history(history : pd.DataFrame = None,
                          metadata : dict = None,
                          figsize=None,
                          print_scores=True,
                          excluded_metrics=[]):
    plt.figure(figsize=figsize)
    metrics = [c for c in history.columns if c not in ('epoch', 'step')]
    for metric in metrics:
        if metric not in excluded_metrics:
            plt.plot(history[metric], label=metric)
    if len(metrics) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.show()
    if metadata is not None:
        print(f'training time: {metadata["training time"]:.4f}')
    if print_scores:
        s =  ', '.join(f'Final {metric}: {history[metric].iloc[-1]:.4f}'
                       for metric in metrics)
        print(s)


def print_ml_metrics(model, X, y, label=None):
    # Obtain the predictions
    with torch.no_grad():
        pred = model(torch.Tensor(X))
    # Compute the root MSE
    rmse = np.sqrt(mean_squared_error(y, pred))
    # Compute the MAE
    mae = mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = r2_score(y, pred)
    lbl = '' if label is None else f' ({label})'
    print(f'R2: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}{lbl}')


# ==============================================================================
# DFL Loss Functions (non-differentiable implementations)
# ==============================================================================


class RegretLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        costs = np.sum(c_star * y_hat, axis=-1)
        best_costs = np.sum(c_star * y_star, axis=-1)
        return np.mean(costs - best_costs)

    def __repr__(self):
        res = 'regret'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res



class SelfContrastiveLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        y_hat = get_decisions_(c_hat)
        best_est_costs = np.sum(c_hat * y_hat, axis=-1)
        est_best_costs = np.sum(c_hat * y_star, axis=-1)
        return np.mean(est_best_costs - best_est_costs)

    def __repr__(self):
        res = 'self contrastive'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res



class SPOPlusLoss:
    def __init__(self, smoothing_samples=0, smoothing_std=0.05, seed=42, alpha=2):
        self.smoothing_samples = smoothing_samples
        self.smoothing_std = smoothing_std
        if smoothing_samples > 0:
            self.noise = normal_sample_(mean=0, std=smoothing_std, size=(smoothing_samples, 1, 2), seed=seed)
        else:
            self.noise = np.zeros((1, 1, 2))
        self.alpha = alpha

    def __call__(self, c_star, c_hat, y_star):
        c_hat = c_hat + self.noise
        c_spo = self.alpha * c_hat - c_star
        y_spo = get_decisions_(c_spo)
        best_spo_costs = np.sum(c_spo * y_spo, axis=-1)
        spo_best_costs = np.sum(c_spo * y_star, axis=-1)
        return np.mean(spo_best_costs - best_spo_costs)


        best_est_costs = np.sum(c_hat * y_hat, axis=-1)
        est_best_costs = np.sum(c_hat * y_star, axis=-1)
        return np.mean(est_best_costs - best_est_costs)

    def __repr__(self):
        res = 'SPO+'
        if self.alpha != 2:
            res += f' with alpha={self.alpha}'
        if self.smoothing_samples > 0:
            res += f' (ns={self.smoothing_samples}, std={self.smoothing_std})'
        return res


# ==============================================================================
# Benchmarks
# ==============================================================================


class ProductionProblem(object):
    def __init__(self, values, requirement):
        """TODO: to be defined. """
        # Store the problem configuration
        self.values = values
        self.requirement = requirement

    def solve(self, costs, tlim=None, print_solution=False):
        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the variables
        x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        # Build the requirement constraint
        rcst = slv.Add(sum(values[i] * x[i] for i in range(nv)) >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

        # Set a time limit, if requested
        if tlim is not None:
            slv.SetTimeLimit(1000 * tlim)
        # Solve the problem
        status = slv.Solve()
        # Prepare the results
        if status in (slv.OPTIMAL, slv.FEASIBLE):
            res = []
            # Extract the solution
            sol = [x[i].solution_value() for i in range(nv)]
            res.append(sol)
            # Determine whether the problem was closed
            if status == slv.OPTIMAL:
                res.append(True)
            else:
                res.append(False)
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]

        # Print the solution, if requested
        if print_solution:
            self._print_sol(res[0], res[1], costs)
        return res

    def _print_sol(self, sol, closed, costs):
        # Obtain indexes of selected items
        idx = [i for i in range(len(sol)) if sol[i] > 0]
        # Print selected items with values and costs
        s = ', '.join(f'{i}' for i in idx)
        print('Selected items:', s)
        s = f'Cost: {sum(costs):.2f}, '
        s += f'Value: {sum(self.values):.2f}, '
        s += f'Requirement: {self.requirement:.2f}, '
        s += f'Closed: {closed}'
        print(s)

    def __repr__(self):
        return f'ProductionProblem(values={self.values}, requirement={self.requirement})'



class ProductionProblem2Stage(object):
    def __init__(self, costs, requirement, buffer_cost, integer_vars=True):
        """TODO: to be defined. """
        # Store the problem configuration
        self.costs = costs
        self.requirement = requirement
        self.buffer_cost = buffer_cost
        self.integer_vars = integer_vars

    def solve(self, values, fixed_fsd=None, tlim=None, print_solution=False):
        # Handle the case where values is a single vector
        try:
            values[0][0]
        except:
            values = [values]
        # Quick access to some useful fields
        costs = self.costs
        req = self.requirement
        req_ub = int(np.ceil(req + np.max(np.sum(np.maximum(values, 0), axis=1))))
        bcst = self.buffer_cost
        nv = len(costs)
        ns = len(values) # number of scenario
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the first variables
        if self.integer_vars:
            x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        else:
            x = [slv.NumVar(0, 1, f'x_{i}') for i in range(nv)]
        # Apply first-stage decisions (if specified)
        if fixed_fsd is not None:
            for i in range(nv):
                slv.Add(x[i] == fixed_fsd[i])
        # Build the second stage variables
        if self.integer_vars:
            y = [slv.IntVar(0, req_ub, f'y_{k}') for k in range(ns)]
        else:
            y = [slv.NumVar(0, req_ub, f'y_{k}') for k in range(ns)]
        # Build the requirement constraint (for every scenario)
        for k in range(ns):
            slv.Add(sum(values[k][i] * x[i] for i in range(nv)) + y[k] >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)) + bcst/ns * sum(y[k] for k in range(ns)))
        # Set a time limit, if requested
        if tlim is not None:
            slv.SetTimeLimit(1000 * tlim)
        # Solve the problem
        status = slv.Solve()
        # Prepare the results
        if status in (slv.OPTIMAL, slv.FEASIBLE):
            res = []
            # Extract the solution
            sol = [x[i].solution_value() for i in range(nv)]
            res.append(sol)
            # Determine whether the problem was closed
            if status == slv.OPTIMAL:
                res.append(True)
            else:
                res.append(False)
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]
        # Print the solution, if requested
        if print_solution:
            self._print_sol(res[0], res[1], res[2],
                      values, [y[k].solution_value() for k in range(ns)])
        return res

    def rel_evpf(self, sol, true_values, tlim=None):
        # Compute the expected cost of the solution
        sol, _, obj, _  = self.solve(true_values, fixed_fsd=sol, tlim=tlim)
        # For every scenario, compute the perfect information solution
        pf_objs = []
        for tv in true_values:
            _, _, pf_obj, _  = self.solve([tv], tlim=tlim)
            pf_objs.append(pf_obj)
        pf_objs = np.array(pf_objs)
        # Compute the average regret w.r.t. the perfect information solutions
        return np.mean((obj - pf_objs) / pf_objs)

    def rel_regret(self, sol, true_values, tlim=None):
        # Compute the expected cost of the solution
        sol, _, obj, _  = self.solve(true_values, fixed_fsd=sol, tlim=tlim)
        # Compute the best possibe solution (assuming a perfect sample)
        tsol, _, tobj, _  = self.solve(true_values, tlim=tlim)
        # Compute the average regret w.r.t. the perfect information solutions
        return (obj - tobj) / np.abs(tobj)

    def _print_sol(self, sol, closed, obj, values, y_vals):
        # Obtain indexes of selected items
        idx = [i for i in range(len(sol)) if sol[i] > 0]
        # Print selected items with values and costs
        s = ', '.join(f'{i}' for i in idx)
        print('Selected items:', s)
        s = f'Cost: {obj:.2f}, '
        s += f'Recourse Actions: {y_vals}, '
        s += f'Requirement: {self.requirement:.2f}, '
        s += f'Closed: {closed}'
        print(s)

    def __repr__(self):
        return f'ProductionProblem2Stage(costs={self.costs}, requirement={self.requirement}, buffer_cost={self.buffer_cost})'



def generate_problem(nitems, rel_req, seed=None, surrogate=False):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item values
    values = 1 + 0.4*np.random.rand(nitems)
    # Generate the requirement
    req = rel_req * np.sum(values)
    # Return the results
    if not surrogate:
        return ProductionProblem(values, req)
    else:
        return ProductionProblemSurrogate(values, req)



def generate_costs(nsamples, nitems, noise_scale=0,
                   seed=None, sampling_seed=None,
                   nsamples_per_point=1,
                   noise_type='normal', noise_scale_type='absolute'):
    assert(noise_scale >= 0)
    assert(nsamples_per_point > 0)
    assert(noise_type in ('normal', 'rayleigh'))
    assert(noise_scale_type in ('absolute', 'relative'))
    # Seed the RNG
    np.random.seed(seed)
    # Generate costs
    # speed = np.random.choice([-14, -11, 11, 14], size=nitems)
    speed = np.random.choice([-20, -10, 10, 20], size=nitems)
    scale = 1.3 + 1.3 * np.random.rand(nitems)
    base = 1 * np.random.rand(nitems) / scale
    offset = -0.75 + 0.5 * np.random.rand(nitems)

    # Generate input
    if sampling_seed is not None:
        np.random.seed(sampling_seed)
    x = np.random.rand(nsamples)
    x = np.repeat(x, nsamples_per_point)

    # Prepare a result dataset
    res = pd.DataFrame(data=x, columns=['x'])

    # scale = np.sort(scale)[::-1]
    for i in range(nitems):
        # Compute base cost
        cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
        # Rebase
        cost = cost - np.min(cost) + base[i]
        # sx = direction[i]*speed[i]*(x+offset[i])
        # cost = base[i] + scale[i] / (1 + np.exp(sx))
        res[f'C{i}'] = cost
    # Add noise
    if noise_scale > 0:
        for i in range(nitems):
            # Define the noise scale
            if noise_scale_type == 'absolute':
                noise_scale_vals = noise_scale * res[f'C{i}']**0
            elif noise_scale_type == 'relative':
                noise_scale_vals = noise_scale * res[f'C{i}']
            # Define the noise distribution
            if noise_type == 'normal':
                noise_dist = stats.norm(scale=noise_scale_vals)
            elif noise_type == 'rayleigh':
                # noise_dist = stats.expon(scale=noise_scale_vals)
                noise_dist = stats.rayleigh(scale=noise_scale_vals)
            r_mean = noise_dist.mean()
            # pnoise = noise * np.random.randn(nsamples)
            # res[f'C{i}'] = res[f'C{i}'] + pnoise
            pnoise = noise_dist.rvs()
            res[f'C{i}'] = res[f'C{i}'] + pnoise - r_mean
    # Reindex
    res.set_index('x', inplace=True)
    # Sort by index
    res.sort_index(inplace=True)
    # Normalize
    vmin, vmax = res.min().min(), res.max().max()
    res = (res - vmin) / (vmax - vmin)

    # Return results
    return res


def generate_2s_problem(nitems, requirement, rel_buffer_cost,
                                seed=None, integer_vars=True):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item costs
    costs = 1 + 0.4*np.random.rand(nitems)
    # Generate the cost of the buffer product
    buffer_cost = rel_buffer_cost * np.mean(costs)
    # Return the results
    return ProductionProblem2Stage(costs, requirement, buffer_cost,integer_vars=integer_vars)



# ==============================================================================
# Models and training code
# ==============================================================================

# def build_ml_model(input_size, output_size, hidden=[],
#         output_activation='linear', scale=None, name=None):
#     # Prepare data structure to store all sizes and layers
#     sizes = [input_size] + hidden
#     layers = []
#     # Build all the hidden layers
#     for s_in, s_out in zip(sizes[:-1], sizes[1:]):
#         layers.append(nn.Linear(s_in, s_out))
#         layers.append(nn.ReLU())
#     # Build the output layer
#     layers.append(nn.Linear(sizes[-1], output_size))

class TwoPhaseModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_sizes=[],
                 name='model', lr=1e-3):
        super().__init__()

        # Store the model parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.lr = lr

        # Prepare data structure to store all sizes and layers
        sizes = [input_size] + hidden_sizes
        layers = []
        # Build all the hidden layers
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nn.ReLU())
        # Build the output layer
        layers.append(nn.Linear(sizes[-1], output_size))
        # Build the model object
        self.model = nn.Sequential(*layers)

        # Save hyperparameters so they can be retrieved from checpoints
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = nn.functional.mse_loss(y_hat, y)
    #     # self.log('val_loss', loss, on_step=False, on_epoch=True)
    #     self.log('val_loss', loss)
    #     return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
        # return optim.Muon(self.parameters(), lr=0.001)
        # return optim.AdamW(self.parameters(), lr=0.001)


def train_ml_model(model, X, y, epochs=20,
        verbose=0, patience=10, batch_size=32,
        load_cached=False, accelerator='auto',
        enable_progress_bar=True):
    # Some fixed parameters
    logdir = 'logs'
    vrs = 'v0'
    cpdir = os.path.join(logdir, model.name, vrs, 'checkpoints')
    metric_log_fname = os.path.join(logdir, model.name, vrs, 'metrics.csv')
    metadata_fname = os.path.join(logdir, model.name, 'metadata.json')

    # Handle model loading
    # NOTE this contains an early exit
    if load_cached and os.path.isdir(cpdir):
        # Load the model weights from the checkpoint
        cpfiles = [f for f in os.listdir(cpdir) if f.endswith('ckpt')]
        if len(cpfiles) == 0:
            raise RuntimeError('No checkpoint found')
        if len(cpfiles) > 1:
            raise RuntimeError('Multiple checkpoints found')
        cpfname = cpfiles[0]
        model = model.__class__.load_from_checkpoint(os.path.join(cpdir, cpfname))
        # Load the metrics
        history = pd.read_csv(metric_log_fname)
        # Load the metadata
        with open(metadata_fname) as fp:
            metadata_dict = json.load(fp)
        return model, history, metadata_dict

    # Wrap the training data
    dset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dloader = DataLoader(dset, batch_size=batch_size)

    # Train
    checkptcb = ModelCheckpoint(os.path.join(logdir, model.name, vrs, 'checkpoints'),
                                enable_version_counter=False)
    logger = CSVLogger(logdir, model.name, version=vrs)
    trainer = pl.Trainer(max_epochs=epochs, logger=logger,
                        enable_model_summary=False,
                        accelerator=accelerator,
                        enable_progress_bar=enable_progress_bar,
                        callbacks=[checkptcb],
                        log_every_n_steps=np.ceil(len(X) / batch_size))
    train_start = time.time()
    trainer.fit(model, dloader)
    train_time = time.time() - train_start

    # Store the metadata in the logging directory
    metadata_dict = {'training time': train_time}
    try:
        metadata_dict['solution count'] = {str(k): v for k, v in model.get_solcount(by_subgroup=True).items()}
    except AttributeError:
        pass
    with open(metadata_fname, 'w') as fp:
        json.dump(metadata_dict, fp)

    # Read the log and retrive the metric curves
    history = pd.read_csv(metric_log_fname)
    return model, history, metadata_dict


def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
    # Obtain all predictions
    with torch.no_grad():
        costs = predictor(torch.Tensor(pred_in)).numpy()
    # Compute all solutions
    sols = []
    for c in costs:
        sol, _, _, _  = problem.solve(c, tlim=tlim)
        sols.append(sol)
    sols = np.array(sols)
    # Compute the true solutions
    tsols = []
    for c in true_costs:
        sol, _, _, _ = problem.solve(c, tlim=tlim)
        tsols.append(sol)
    tsols = np.array(tsols)
    # Compute true costs
    costs_with_predictions = np.sum(true_costs * sols, axis=1)
    costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
    # Compute regret
    regret = (costs_with_predictions - costs_with_true_solutions) / np.abs(costs_with_true_solutions)
    # Return true costs
    return regret


def compute_evpf_2s(problem, predictor, data_ts, tlim=None):
    # Obtain all inputs
    pred_in = np.unique(data_ts.index).reshape(-1, 1)
    # Compute all predictions
    with torch.no_grad():
        pred_values = predictor(torch.Tensor(pred_in)).numpy()
    # Compute the EVPF for every input
    evpfs = []
    for pin, pvals in zip(pred_in, pred_values):
        # Compute the solution
        sol, closed, obj, wtime = problem.solve([pvals], tlim=tlim)
        # Obtain the true value
        true_values = data_ts.loc[pin].values
        # Compute the EVPF for this solution
        evpf = problem.rel_evpf(sol, true_values, tlim=tlim)
        evpfs.append(evpf)
    # Return results
    return np.array(evpfs)


class DFLModel(TwoPhaseModel):

    _supported_methods = ('spo', 'sfge_2s', 'bonis_2s', 'bofis_2s')

    def __init__(self, input_size, output_size, problem,
                 hidden_sizes=[],
                 name='model',
                 lr=1e-3,
                 dfl_method='spo',
                 dfl_tlim=None,
                 dfl_params={},
                 dfl_max_workers=None,
                 warm_start_state_dict=None,
                 drop_cache_at_end=False):
        if dfl_method not in DFLModel._supported_methods:
            raise ValueError(f'Unsupported DFL method {dfl_method}. Use one from {DFLModel._supported_methods}')

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         hidden_sizes=hidden_sizes,
                         name=name,
                         lr=lr)
        # Warm stat, if an initial state is provided
        if warm_start_state_dict is not None:
            self.load_state_dict(warm_start_state_dict)

        # Store the DFL-specific parameters
        self.prb = problem
        self.method = dfl_method
        self.method_params = dfl_params
        self.tlim = dfl_tlim
        self.max_workers = dfl_max_workers

        # Additional learnable parameters
        if dfl_method in ('sfge_2s', 'bonis_2s', 'bofis_2s'):
            if 'sigma' in dfl_params:
                self.sigma = dfl_params['sigma']
            else:
                self.sigma = 0.1
        if dfl_method in ('bonis_2s', 'bofis_2s'):
            # Sigma multiplier
            if 'sigma_multiplier' in dfl_params:
                self.sigma_mult = dfl_params['sigma_multiplier']
            else:
                self.sigma_mult = 1.1 # 10% more standard deviation
        if dfl_method == 'bofis_2s':
            # Max ratio
            if 'max_iw_ratio' in dfl_params:
                self.max_iw_ratio = dfl_params['max_iw_ratio']
            else:
                self.max_iw_ratio = 2

        # Prepare some state objects
        self.cache = self.DFLCacheManager(drop_cache_at_end)

        # Save hyperparameters so they can be retrieved from checpoints
        self.save_hyperparameters()

    # Cache manager section
    # -----------------------------------------------------------------------
    CacheEntry = namedtuple('CacheEntry', ['sol', 'closed', 'obj', 'soltime', 'mu', 'idx'])

    class DFLCacheManager(Callback):
        """This callback resets the cache before and after training"""
        def __init__(self, drop_cache_at_end):
            self.drop_cache_at_end = drop_cache_at_end
            self.reset()

        # Cache-related methods and classes
        def reset(self):
            self.cache = {}
            self.last_idx = 0

        def _get_keys(self, x, y, y_hat):
            kv = tuple(x.tolist() + y.tolist())
            iv = tuple(y_hat.tolist()) if y_hat is not None else None
            return kv, iv

        def lookup(self, x, y, y_hat=None):
            # Prepare the results
            res = None
            # Convert inputs to keys
            kv, iv = self._get_keys(x, y, y_hat)
            # Main lookup (by context info)
            if kv in self.cache:
                if iv is not None:
                    # Mode 1: specific solution lookup
                    if iv in self.cache[kv]:
                        res = self.cache[kv][iv]
                    else:
                        return None
                else:
                    # Mode 2: all solution lookup
                    res = self.cache[kv]
            return res

        def store(self, x, y, y_hat, sol, closed, obj, soltime, mu=None):
            # Convert inputs to keys
            kv, iv = self._get_keys(x, y, y_hat)
            # Build the first-level cache, if missing
            if kv not in self.cache:
                self.cache[kv] = {}
            self.cache[kv][iv] = DFLModel.CacheEntry(sol, closed, obj, soltime,
                                                     mu=mu,
                                                     idx=self.last_idx)
            self.last_idx += 1

        def on_train_start(self, trainer, pl_module):
            # Reset the cache
            self.reset()

        def on_train_end(self, trainer, pl_module):
            # Reset the cache
            if self.drop_cache_at_end:
                self.reset()

        def get_solcount(self, by_subgroup=False):
            if by_subgroup:
                res = {}
                for k, subcache in self.cache.items():
                    res[k] = len(subcache)
            else:
                res = 0
                for subcache in self.cache.values():
                    res += len(subcache)
            return res

    # Register the cache manager
    def configure_callbacks(self):
        return [self.cache]

    # Get the solution count
    def get_solcount(self, by_subgroup=False):
        return self.cache.get_solcount(by_subgroup)

    def _retrieve_or_compute_one(prb, cache,
                                 xv, yv, yv_hat,
                                 tlim, method,
                                 mu=None):
        with torch.no_grad():
            # Attempt a cache lookup
            res = cache.lookup(xv, yv, yv_hat)
            if res is not None:
                # In case of a cache hit, just return the solution
                sol = res.sol
                obj = res.obj
            else:
                # In case of a cache miss, compute a solution
                sol, closed, _, soltime = prb.solve(yv_hat.tolist(), tlim=tlim)
                # True cost evalution
                if method == 'spo':
                    obj = np.dot(yv, sol)
                elif method in ('sfge_2s', 'bonis_2s', 'bofis_2s'):
                    _, _, obj, _ = prb.solve(yv.tolist(), fixed_fsd=sol, tlim=tlim)
                else:
                    raise ValueError(f'Invalid DFL method during cost evaluation: {method}')
                # ...Then store it in the cache
                cache.store(xv, yv, yv_hat, sol, closed, obj, soltime, mu)
        # Return the results
        return sol, obj

    def _retrieve_or_compute_batch(self, x, y, y_hat, mu=None):
        # Prepare a result data structure
        res_sol = [None] * len(x)
        res_obj = [None] * len(x)
        # Prepare optional input data structures
        if mu is None:
            mu = [None] * len(x)
        # Process all (x, y, y_hat) triplets pairs
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start all jobs
            dfl_futures = {executor.submit(DFLModel._retrieve_or_compute_one,
                                           self.prb, self.cache,
                                           xv, yv, yv_hat,
                                           self.tlim, self.method,
                                           muv) : pos
                           for pos, (xv, yv, yv_hat, muv)
                                in enumerate(zip(x, y, y_hat, mu))}
            # Collect the results
            for ft in as_completed(dfl_futures):
                pos = dfl_futures[ft]
                sol, obj = ft.result()
                res_sol[pos] = sol
                res_obj[pos] = obj
        # Return the results
        return torch.Tensor(res_sol), torch.Tensor(res_obj)


    def _retrieve_last_cached(self, x, y, max_res=None):
        # Prepare a result data structure
        res_yhat, res_obj, res_mu = [], [], []
        # Get all cached samples
        all_cached = self.cache.lookup(x, y)
        # Determine the number of results
        num = len(all_cached)
        if max_res is not None:
            num = min(num, max_res)
        # If more then one solution is found, sort them by decreasing idx
        lkey = lambda kv: kv[1].idx
        for yhat, cres in heapq.nlargest(num,
                                         all_cached.items(),
                                         key=lkey):
            res_yhat.append(yhat)
            res_obj.append(cres.obj)
            res_mu.append(cres.mu)
        return torch.Tensor(res_yhat), torch.Tensor(res_obj), torch.Tensor(res_mu)


    def _compute_spo_loss_terms(self, x, y, y_hat):
        # Retrieve the reference (clairvoyant) solutions
        refsols, refobjs = self._retrieve_or_compute_batch(x, y, y)
        # Compute the best solutions w.r.t. the predictions
        sols, objs = self._retrieve_or_compute_batch(x, y, y_hat)
        # Compute the SPO costs
        spo_costs = 2 * y_hat - y
        loss_terms = torch.sum(spo_costs * (refsols - sols), dim=1)
        # Compute the current regrets (for logging)
        regrets = objs - refobjs
        return loss_terms, regrets


    def _compute_sfge_loss_terms(self, x, y, y_hat):
        # Build the smoothing distribution
        sigma_ref = torch.eye(self.output_size) * self.sigma**2
        dists = torch.distributions.MultivariateNormal(loc=y_hat,
                                                       covariance_matrix=sigma_ref)

        with torch.no_grad():
            # Compute the samples
            samples = dists.sample()

            # Retrieve or compute the reference (clairvoyant) solutions
            _, refobjs = self._retrieve_or_compute_batch(x, y, y)

            # Retrieve or compute the solution associated to the current samples
            _, objs = self._retrieve_or_compute_batch(x, y, samples)

            # Compute regrets
            regrets = objs - refobjs

            # Regret standardization
            advantage = (regrets - torch.mean(regrets)) / torch.std(regrets)

        # Compute the sample log probabilities
        log_probs = dists.log_prob(samples)
        return log_probs * advantage, regrets


    def _compute_bonis_loss_terms(self, x, y, y_hat):
        # Build the smoothed distribution objectes
        sigma_ref_val = self.sigma**2
        sigma_ref = torch.eye(self.output_size) * sigma_ref_val
        dists_ref = torch.distributions.MultivariateNormal(loc=y_hat,
                                                       covariance_matrix=sigma_ref)

        # Build the sampling distribution objects
        sigma_smp_val = sigma_ref_val * self.sigma_mult**2
        sigma_smp = torch.eye(self.output_size) * sigma_smp_val
        dists_smp = torch.distributions.MultivariateNormal(loc=y_hat,
                                                       covariance_matrix=sigma_smp)
        with torch.no_grad():
            # Retrieve or compute the reference (clairvoyant) solutions
            _, refobjs = self._retrieve_or_compute_batch(x, y, y)

            # Compute the samples
            samples = dists_smp.sample()
            # Retrieve or compute the solution associated to the current samples
            _, objs = self._retrieve_or_compute_batch(x, y, samples)

            # Compute regrets
            regrets = objs - refobjs

            # Compute advantage
            advantage = (regrets - torch.mean(regrets)) / torch.std(regrets)

            # Compute the importance weights denominators
            iwgt_den_log = dists_smp.log_prob(samples)

        # Compute the numerator of the importance weights
        iwgt_num_log = dists_ref.log_prob(samples)
        # Compute the importance weights
        iwgt = (iwgt_num_log - iwgt_den_log).exp()
        # Return the loss terms
        return iwgt * advantage, regrets


    def _compute_bofis_loss_terms_one(self, x, y, y_hat):
        # Build the smoothed distribution object
        sigma_ref_val = self.sigma**2
        sigma_ref = torch.eye(self.output_size) * sigma_ref_val
        dist_ref = torch.distributions.MultivariateNormal(loc=y_hat,
                                                       covariance_matrix=sigma_ref)

        with torch.no_grad():
            # Build the sampling distribution object
            sigma_smp_val = sigma_ref_val * self.sigma_mult**2
            sigma_smp = torch.eye(self.output_size) * sigma_smp_val
            dist_smp = torch.distributions.MultivariateNormal(loc=y_hat,
                                                           covariance_matrix=sigma_smp)


            # Retrieve or compute the reference (clairvoyant) solutions
            _, refobj = DFLModel._retrieve_or_compute_one(self.prb, self.cache,
                                                       x, y, y,
                                                       self.tlim, self.method,
                                                       mu=y.tolist())


            # Retrieve all cached samples
            cached_yhat, cached_obj, cached_mu = self._retrieve_last_cached(x, y,
                                                                         max_res=1)
            # Build the past sample distributions
            cached_dists = torch.distributions.MultivariateNormal(loc=cached_mu,
                                                          covariance_matrix=sigma_smp)

            # Compute the log densities of the current smoothing center w.r.t. all of
            # the cached distribution
            cached_logprob = cached_dists.log_prob(y_hat)
            if cached_logprob.size()[0] > 1:
                raise Exception('THIS SHOULD NOT HAPPEN (YET)')
                # Adjust the log probs to account for the distribution weights
                n_cached = cached_mu.size()[0]
                cached_logprob_weighted = cached_logprob + np.log(1 / n_cached)
                # Apply weights and obtain a single log prob
                cached_logprob = torch.logsumexp(cached_logprob_weighted)

            # Compute the density of the center w.r.t. the smoothing distribution
            current_logprob = dist_ref.log_prob(y_hat)

            # Compare the density ratio with the allowed maximum
            thr = np.log(self.max_iw_ratio)
            if current_logprob - cached_logprob > thr:
                # Sample the smoothing distribution
                sample = dist_smp.sample()
                # Retrieve or compute the solution associated to the current samples
                _, obj = DFLModel._retrieve_or_compute_one(self.prb, self.cache,
                                                           x, y, sample,
                                                           self.tlim, self.method,
                                                           mu=y_hat.tolist())
                # Compute the importance weight denominator
                iwgt_den_log = dist_smp.log_prob(sample)
            else:
                # Retrieve cached predictions and objective
                sample = cached_yhat[0]
                obj = cached_obj
                # Compute the importance weight denominator
                iwgt_den_log = cached_dists.log_prob(sample)[0]

            # Compute the regret
            regret = obj - refobj

        # Compute the numerator of the importance weights
        iwgt_num_log = dist_ref.log_prob(sample)
        # Compute log importance weight
        iwgt_log = iwgt_num_log - iwgt_den_log
        # Return the results
        return iwgt_log, regret


    def _compute_bofis_loss_terms(self, x, y, y_hat):
        # Prepare a result data structure
        res_iwgts, res_regrets = [], []
        # Process all (x, y, y_hat) triplets pairs
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start all jobs
            dfl_futures = [executor.submit(DFLModel._compute_bofis_loss_terms_one,
                                           self,
                                           xv, yv, yv_hat)
                           for xv, yv, yv_hat in zip(x, y, y_hat)]
            # Collect the results
            for ft in as_completed(dfl_futures):
                term, regret = ft.result()
                res_iwgts.append(term)
                res_regrets.append(regret)

        # Convert the results to tensors
        iwgts = torch.stack(res_iwgts).exp()
        regrets = torch.Tensor(res_regrets)
        # Compute advantage
        advantage = (regrets - torch.mean(regrets)) / torch.std(regrets)
        # Return the loss terms
        return iwgts * advantage, regrets


    def _compute_loss_terms(self, x, y, y_hat):
        # Compute the actual loss terms
        if self.method == 'spo':
            loss_terms, regrets = self._compute_spo_loss_terms(x, y, y_hat)
        elif self.method == 'sfge_2s':
            loss_terms, regrets = self._compute_sfge_loss_terms(x, y, y_hat)
        elif self.method == 'bonis_2s':
            loss_terms, regrets = self._compute_bonis_loss_terms(x, y, y_hat)
        elif self.method == 'bofis_2s':
            loss_terms, regrets = self._compute_bofis_loss_terms(x, y, y_hat)
        else:
            raise Exception(f'Invalid DFL method "{self.method}"')
        return loss_terms, regrets

    # DFL training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Obtain the preditions
        y_hat = self(x)
        # Compute the loss terms
        loss_terms, regrets = self._compute_loss_terms(x, y, y_hat)
        # Compute the loss function
        loss = torch.mean(loss_terms)
        # Compute the average regret (just for logging)
        regret = torch.mean(regrets)
        # Logging
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('regret', regret, on_step=False, on_epoch=True)
        return loss
