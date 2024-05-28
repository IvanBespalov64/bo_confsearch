#import calc # For init CURRENT_STRUCTURE_ID

from transform_kernel import TransformKernel

from grad_gpr import GPRwithGrads

from coef_from_grid import pes, pes_tf, pes_tf_grad

from calc import calc_energy, load_last_optimized_structure_xyz_block
from calc import change_dihedrals
from calc import parse_points_from_trj
from calc import load_params_from_config

from coef_calc import CoefCalculator

from db_connector import LocalConnector

from dataclasses import fields

from sparse_ei import SparseExpectedImprovement
from sparse_grad_based_ei import SparseGradExpectedImprovement
from explor_imp import ExplorationImprovement
from imp_var import ImprovementVariance

import trieste
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.data import Dataset
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from mean_cos import BaseCos, BaseCosMod
import plotly.graph_objects as go
from scipy.special import erf
import sys
import os
import yaml
import json

from default_vals import ConfSearchConfig

from trieste.acquisition.function import ExpectedImprovement

from rdkit import Chem

from sklearn.cluster import DBSCAN

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

MOL_FILE_NAME = None
NORM_ENERGY = 0.

DIHEDRAL_IDS = []

CUR_ADD_POINTS = []

global_degrees = []

structures_path = ""
exp_name = ""

ASKED_POINTS = []

model_chk = None
current_minima = 1e9
acq_vals_log = []

LAST_OPT_OK = True

def degrees_to_potentials(
    degrees : np.ndarray,
    mean_func_coefs : np.ndarray
) -> np.ndarray:
    return [
        [pes(np.array([[degree[i]]]), *coefs[i])[0] for i in range(len(degree))]
        for degree in degrees
    ]  

# defines a functions, that will convert args for mean_function
def parse_args_to_mean_func(inp):
    """
    Convert input data from [inp_dim, 7] to [7, inp_dim]
    """
    return tf.transpose(inp)

def calc(dihedrals : list[float]) -> float:
    """
        Perofrms calculating of energy with current dihedral angels
    """
    
    def dump_status_hook(
        dumping_value : bool,
        filename : str = exp_name+"_last_opt_status.json"
    ) -> None:
        with open(filename, 'w') as file:
            json.dump({
                "LAST_OPT_OK" : dumping_value
            }, file)

    global LAST_OPT_OK

    if model_chk:
        print(f"Checkpoint is not null, calculating previous acq. func. max!")
        dihedrals_tf = tf.constant(dihedrals, dtype=tf.float64)
        if len(dihedrals_tf.shape) == 1:
            dihedrals_tf = tf.reshape(dihedrals_tf, [1, dihedrals_tf.shape[0]])
        print(f"Cur dihedrals_tf: {dihedrals_tf}")
        print(f"Current minima: {current_minima}")
        mean, variance = model_chk.predict_f(dihedrals_tf)
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        tau = current_minima + 3.
        acq_val = normal.cdf(tau) * (((tau - mean)**2) * (1 - normal.cdf(tau)) + variance) + tf.sqrt(variance) * normal.prob(tau) *\
                (tau - mean) * (1 - 2*normal.cdf(tau)) - variance * (normal.prob(tau)**2)
        print(f"Previous acq. val: {acq_val}")
        acq_vals_log.append(acq_val.numpy().flatten()[0])        

    if tf.is_tensor(dihedrals):
        dihedrals = list(dihedrals.numpy())

    ASKED_POINTS.append(dihedrals)
    
    print(f"Point: {dihedrals}")

    #Pre-opt
    print('Optimizing constrained struct')
    en, preopt_status = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, constrained_opt=True)
    LAST_OPT_OK = preopt_status
    print(f"Status of preopt: {preopt_status}; LAST_OPT_OK: {LAST_OPT_OK}")
    if not preopt_status:
        dump_status_hook(dumping_value=LAST_OPT_OK)
        return en + np.random.randn()
    print('Optimized!\nLoading xyz from preopt')
    xyz_from_constrained = load_last_optimized_structure_xyz_block(MOL_FILE_NAME)
    print('Loaded!\nFull opt')
    en, opt_status = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, force_xyz_block=xyz_from_constrained)
    LAST_OPT_OK = opt_status
    print(f"Status of opt: {opt_status}; LAST_OPT_OK: {LAST_OPT_OK}")
    print(f'Optimized! En = {en}')
    dump_status_hook(dumping_value=LAST_OPT_OK)
    return en + ((not opt_status) * np.random.randn())

def max_comp_dist(x1, x2, period : float = 2 * np.pi):
    """
        Returns dist between two points:
        d(x1, x2) = max(min(|x_1 - x_2|, T - |x_1 - x_2|))), 
        where T is period
    """
    
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if not isinstance(x2, np.ndarray):
        x2 = np.array(x2)
        
    return np.max(np.min((np.abs(x1 - x2), period - np.abs(x1 - x2)), axis=0))

def save_res(xyz_name : str, a : float, b : float):
    """
        saves to 'xyz_name' geometry with a and b
    """
    with open(xyz_name, 'w+') as file:
        file.write(change_dihedrals(MOL_FILE_NAME, [([1, 2, 3, 4], a), ([0, 1, 2, 3], b)], True))

def save_all(all_file_name : str, points : list):
    """
        saves all structures
    """
    with open(all_file_name, 'w+') as file:
        for cur in points:
	        a, b = cur
	        file.write(change_dihedrals(MOL_FILE_NAME, [([1, 2, 3, 4], a), ([0, 1, 2, 3], b)], True))
           

def save_plot(plot_name : str, points : list, z : list, plotBorder = True):
    """
        saves plot with points
    """
    print("Enter")
    fig = go.Figure()
    x, y = [], []
    z = z.reshape(len(z), )
    print("Init")
    for cur in points:
        a, b = cur
        x.append(a)
        y.append(b)
    fig = go.Figure(data=[go.Scatter3d(x = x, y = y, z = z,
                                   mode='markers',
                                   text = [_ for _ in range(1, len(points) + 1)])])
    fig.write_html(plot_name)
    if plotBorder:
        r = np.linspace(0, 2 * np.pi, 100)
        border = np.ones((100, 100)) * (np.min(z) + 3.)
        fig.add_trace(go.Surface(x = r, y = r, z = border))
        fig.write_html("tests/bordered.html")
    

def save_acq(file_name : str, 
             model : gpflow.models.gpr.GPR,
             points : list,
             vals : list):
    """
        Saves plot of ExpectedImporovement acquisitioon function
    """
    x, y = zip(*points)
   
    left_borders, right_borders = roi_calc(model, points)
 
 
    fig = go.Figure()
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    pred_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T

    cur_minimum = np.min(vals)

    points = np.array(points)
    vals = np.array(vals)

    mean, variance = model.predict_f(pred_points)
 
    left_borders, right_borders = roi_calc(model, MINIMA)

    #grads = model.predict_grad(tf.constant(points, dtype=tf.float64))
    #print(grads.numpy())

    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    acq_vals =  (cur_minimum - mean) * normal.cdf(cur_minimum) + variance * normal.prob(cur_minimum)
    rdists = tf.math.mod(tf.abs(tf.constant(pred_points.reshape(len(pred_points), 1, 2)) - tf.constant(points.reshape(1, len(points), 2))), 2 * np.pi)
    dists = tf.reshape(tf.reduce_min(
                   tf.math.reduce_max(tf.minimum(rdists, 2*np.pi - rdists), axis=-1), 
                                axis=-1), [pred_points.shape[0], 1])
    
    acq_numpy = acq_vals.numpy()
    plot_acq = []
    for i in range(len(acq_numpy)): 
        plot_acq.append(is_unique(pred_points[i], MINIMA, left_borders, right_borders, hide_asked_points=False) * acq_numpy[i])

    #fig.add_surface(x=xx, y=yy, z=acq_vals.numpy().reshape((30, 30)))
    fig.add_surface(x=xx, y=yy, z=np.array(plot_acq).reshape((30, 30)))
    fig.write_html(file_name)

def roi_calc(
    model : GPRwithGrads,
    points : list
) -> list:
    """
            Gets model with grad supporing and observed points
            Calculates Regions of Interest (borders where gradient changes the sign)
    """
    def equal(
        a : float, 
        b : float,
        eps : float = 1e-6
    ) -> bool:
        return np.abs(a - b) < eps
    
    dims = len(points[0])
    right_borders = [(np.pi / 12) * np.ones(dims) for j in range(len(points))]
    left_borders = [(np.pi / 12) * np.ones(dims) for j in range(len(points))]

    for idx, cur in enumerate(points):
        cur = np.array(cur)

        for dim in range(dims):
            cur_mask = np.zeros(dims)
            cur_mask[dim] = np.pi / 12
            right_border_init_grad = model.predict_grad(tf.constant([(cur + cur_mask) % (2*np.pi)], dtype=tf.float64)).numpy().flatten()[dim]
            left_border_init_grad = model.predict_grad(tf.constant([(cur - cur_mask + 2*np.pi) % (2*np.pi)], dtype=tf.float64)).numpy().flatten()[dim]
            left_flag, right_flag = False, False
            for i in range(2, 8):
                if not right_flag:
                    right_border_grad = model.predict_grad(tf.constant([(cur + i * cur_mask) % (2*np.pi)], dtype=tf.float64)).numpy().flatten()[dim]  
                if not left_flag:
                    left_border_grad = model.predict_grad(tf.constant([(cur - i * cur_mask + 2*np.pi) % (2*np.pi)], dtype=tf.float64)).numpy().flatten()[dim]
                if not right_flag and (right_border_grad*right_border_init_grad < 0):
                    right_borders[idx][dim] = i*np.pi/12
                    right_flag = True
                if not left_flag and (left_border_grad*left_border_init_grad < 0):
                    left_borders[idx][dim] = i*np.pi/12
                    left_flag = True
                if right_flag and left_flag:
                    continue
    return left_borders, right_borders


def save_prob(file_name : str, model : gpflow.models.gpr.GPR, points : list, vals = None, mean_func_coefs : np.ndarray = None):
    """
        Saves max prob and plots last GP
    """
    fig = go.Figure()
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
   
    xx_g, yy_g = np.meshgrid(xx, yy)
    predicted_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T
    mean, var = model.predict_f(predicted_points)
    
    left_borders, right_borders = roi_calc(model, MINIMA)
    print(f"MINIMA: {MINIMA}") 
    #print(f"Var: {var}\nMean: {mean}")

    print(f"Min var = {np.min(var)}, max var = {np.max(var)}")
    cur_minima = np.min(vals)
    prob = 0.5 * (erf((cur_minima + 3. - mean) / np.sqrt(2 * var)) + 1)
    
    max_unknown_prob = 0.
    plot_points = []
    plot_prob = []
    for i in range(len(mean)):
        plot_points.append(predicted_points[i])
        #plot_prob.append(is_unique(predicted_points[i], MINIMA, left_borders, right_borders) * prob[i])
        plot_prob.append(is_unique(predicted_points[i], points, left_borders, right_borders) * prob[i])
    fig.add_trace(go.Surface(x = xx, y = yy, z = np.array(plot_prob).reshape((30, 30)), name='prob')) # plot_prob
    fig.write_html(file_name)
    fig = go.Figure() 
    tau = cur_minima + 3.
    normal = tfp.distributions.Normal(mean, tf.sqrt(var))
    acq_vals = normal.cdf(tau) * (((tau - mean)**2) * (1 - normal.cdf(tau)) + var) + tf.sqrt(var) * normal.prob(tau) *\
                (tau - mean) * (1 - 2*normal.cdf(tau)) - var * (normal.prob(tau)**2)
    acq_vals = acq_vals.numpy()
    
    
    
    #ei = (cur_minima - mean) * normal.cdf(cur_minima) + var * normal.prob(cur_minima) 
    #rdists = tf.math.mod(tf.abs(tf.constant(predicted_points.reshape(len(predicted_points), 1, 2)) - tf.constant(points.reshape(1, len(points), 2))), 2 * np.pi) 
    #dists = tf.reshape(
    #    tf.reduce_min(
    #        tf.math.reduce_max(
    #            tf.minimum(rdists, 2*np.pi - rdists), axis=-1
    #        ),
    #        axis=-1
    #    ), 
    #    [predicted_points.shape[0], 1]
    #).numpy()
    
    fig.add_trace(go.Surface(x = xx, y = yy, z = acq_vals.reshape((30, 30)), name='acq_vals')) 

    #fig.add_trace(go.Surface(x = xx, y = yy, z = np.where(dists > (np.pi/6), ei, 0).reshape((30, 30)), name='acq_vals')) 

    qx, qy = zip(*points)
    fig.add_scatter3d(x = qx, y = qy, z = vals, mode = "markers")
    fig.write_html(file_name.replace('prob_', 'acq_'))

def is_unique(
    cur_point : list,
    points : list,
    left_borders : list,
    right_borders : list,
    hide_asked_points : bool = True           
) -> int:
    """
        Returns 1 if point stands alone in 'threshold'
        in every dimension, 0 otherway
    """  
    
    def in_segment(
        x : float,
        left : float,
        right : float,
        eps : float = 1e-6,
    ) -> bool:
        left = (left + 2*np.pi) % (2*np.pi)
        right = (right + 2*np.pi) % (2*np.pi)
        if left > right:
            return ((x <= (right + eps)) or ((x >= (left - eps)) and (x <= (2*np.pi + eps))))
        return ((x >= (left - eps)) and (x <= (right + eps)))
    
    if hide_asked_points: 
       for asked_point in ASKED_POINTS:
            had_seen = True
            for dim in range(len(cur_point)):
                had_seen = had_seen and in_segment(cur_point[dim], asked_point[dim] - (np.pi/6), asked_point[dim] + (np.pi/6))
            if had_seen:
                return 0        

    for idx, point in enumerate(points):
        had_seen = True
        for dim in range(len(cur_point)):
            #had_seen = had_seen and in_segment(cur_point[dim], point[dim] - left_borders[idx][dim], point[dim] + right_borders[idx][dim])
            had_seen = had_seen and in_segment(cur_point[dim], point[dim] - (np.pi/12), point[dim] + (np.pi/12))
        if had_seen:
            return 0
    return 1

def get_max_unknown_prob(model : gpflow.models.gpr.GPR, points : list, vals : list, step = -1, mean_func_coefs : np.ndarray = None):
    """
        Returns max prob to find another minimum in unknow space
    """
    xx = np.linspace(np.array(points).flatten().min(), np.array(points).flatten().max(), 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    predicted_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T
    mean, var = model.predict_f(predicted_points)
    prob = 0.5 * (erf((np.min(mean) + 3. - mean) / ((2 ** 0.5) * (var ** 0.5))) + 1)
    max_unknown_prob = 0.
    for i in range(len(mean)):
        if is_unique(predicted_points[i], points):
            max_unknown_prob = max(max_unknown_prob, prob[i])
    print("Plotting prob!")
    save_prob(f"probs/prob_{step}.html", model, points, vals, mean_func_coefs)
    print("Plotted!")
    return max_unknown_prob


# defines a function that will be predicted
# cur - input data 'tensor' [n, inp_dim], n - num of points, inp_dim - num of dimensions
def func(cur): 
    return tf.map_fn(fn = lambda x : np.array([calc(x)]), elems = cur)

def upd_points(dataset : Dataset, model : gpflow.models.gpr.GPR) -> tuple[Dataset, gpflow.models.gpr.GPR]:
    """
        update dataset and model from CUR_ADD_POINTS
    """

    degrees, energies = [], []
    for cur in CUR_ADD_POINTS:
        d, e = zip(*cur)
        degrees.extend(d)
        energies.extend(e)
    dataset += Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(energies), 1))
    model.update(dataset)
    model.optimize(dataset)

    return dataset, model

def upd_dataset_from_trj(
    trj_filename : str, 
    dataset : Dataset
) -> Dataset:
    """
        Return dataset that consists of old points
        add points from trj
    """
    print(f"Input dataset is: {dataset}") 
    parsed_data, last_point = parse_points_from_trj(
        trj_file_name=trj_filename, 
        dihedrals=DIHEDRAL_IDS, 
        norm_en=NORM_ENERGY, 
        save_structs=True, 
        structures_path=structures_path, 
        return_minima=True,
    )
    #MINIMA.append(last_point[0]) 
    print(f"Parsed data: {parsed_data}")
    degrees, energies = zip(*parsed_data)
    print(f"Degrees: {degrees}\nEnergies: {energies}")

    global_degrees.extend(degrees) 

    add_part = Dataset(tf.constant(list(degrees), dtype="double"), tf.constant(list(energies), dtype="double").reshape(len(degrees), 1))

    if not dataset:
        return add_part
    else:
        return dataset + add_part

def erase_last_from_dataset(dataset : Dataset, n : int = 1):
    """
        Deletes last n points from trj
    """
    
    query_points = tf.slice(dataset.query_points, [0, 0], [dataset.query_points.shape[0] - n, dataset.query_points.shape[1]])
    observations = tf.slice(dataset.observations, [0, 0], [dataset.observations.shape[0] - n, dataset.observations.shape[1]])

    return Dataset(query_points, observations)

#TODO: Rewrite in tf way
class PotentialFunction():
    def __init__(self, mean_func_coefs) -> None:
        self.mean_func_coefs = mean_func_coefs

    @tf.function
    def __call__(self, X : tf.Tensor) -> tf.Tensor:
        return tf.stack(
                    [
                        pes_tf(X[:, dim], *self.mean_func_coefs[dim]) for dim in range(len(self.mean_func_coefs))
                    ],
                    axis=1
                )
    @tf.function
    def grad(self, X : tf.Tensor) -> tf.Tensor:
        return tf.stack(
                    [
                        pes_tf_grad(X[:, dim], *self.mean_func_coefs[dim]) for dim in range(len(self.mean_func_coefs))
                    ],
                    axis=1
                )

print("Reading config.yaml")

raw_config = {}

try:
    with open('config.yaml', 'r') as file:
        raw_config = yaml.safe_load(file)
except FileNotFoundError:
    print("No config.yaml!\nFinishing!")
    exit(0)
except Exception:
    print("Something went wrong!\nFinishing!")
    exit(0)

config = ConfSearchConfig(**raw_config)

MOL_FILE_NAME = config.mol_file_name
structures_path = config.exp_name + "/"
exp_name = config.exp_name

if not os.path.exists(structures_path):
    os.makedirs(structures_path)

print(f"Performing conf. search with config: {config}")

load_params_from_config({field.name : getattr(config, field.name) for field in fields(config)}) # TODO: rewrite in better way

print("Coef calculator creatring")

coef_matrix = CoefCalculator(
    mol=Chem.RemoveHs(Chem.MolFromMolFile(MOL_FILE_NAME)),
    config=config, 
    dir_for_inps=f"{exp_name}_scans/", 
    db_connector=LocalConnector('dihedral_logs.db')
).coef_matrix()

print("Coef calculator created!")

mean_func_coefs = []

for ids, coefs in coef_matrix:
    DIHEDRAL_IDS.append(ids)
    mean_func_coefs.append(coefs)

print("Dihedral ids", DIHEDRAL_IDS)
print("Mean func coefs", mean_func_coefs)

search_dim = len(DIHEDRAL_IDS)

print("Cur search dim is", search_dim)

amps = np.array([
    np.abs(mean_func_coefs[i][:3]).sum() for i in range(len(mean_func_coefs))
])

potential_func = PotentialFunction(mean_func_coefs)

kernel = gpflow.kernels.White(0.001) + gpflow.kernels.Periodic(gpflow.kernels.RBF(variance=0.07, lengthscales=0.005, active_dims=[i for i in range(search_dim)]), period=[2*np.pi for _ in range(search_dim)]) + TransformKernel(potential_func, gpflow.kernels.RBF(variance=0.12, lengthscales=0.005, active_dims=[i for i in range(search_dim)])) # ls 0.005 var 0.3 -> 0.15

kernel.kernels[1].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64))
kernel.kernels[2].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64))

search_space = Box([0. for _ in range(search_dim)], [2 * np.pi for _ in range(search_dim)])  # define the search space directly

#Calc normalizing energy
#in kcal/mol!

NORM_ENERGY, _ = calc_energy(MOL_FILE_NAME, dihedrals=[], norm_energy=0.)#-367398.19960427243

print(f"Norm energy: {NORM_ENERGY}")

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
dataset = None

for idx in range(config.num_initial_points):
    initial_query_points = search_space.sample_sobol(1)
    observed_point = observer(initial_query_points)
    if not LAST_OPT_OK:
        print(f"Optimization didn't finished well. Continue only with broken_struct_energy in required point: {observed_point}")
        dataset = observed_point if not dataset else dataset + observed_point
    else:
        dataset = upd_dataset_from_trj(f"{MOL_FILE_NAME[:-4]}_trj.xyz", dataset)
    
print(f"Initial dataset observed! {config.num_initial_points} minima observed, total {dataset.query_points.shape[0]} points has been collected!")

#initial_data = observer(initial_query_points)

#MINIMA = initial_data.query_points.numpy().tolist()

#gpr = gpflow.models.GPR(
gpr = GPRwithGrads(
    dataset.astuple(), 
    kernel
)

#print(gpr.parameters)
gpflow.set_trainable(gpr.likelihood, False)
gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)
gpflow.set_trainable(gpr.kernel.kernels[1].period, False)
model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

print(f"Initital data:\n{dataset}")

#dataset = upd_dataset_from_trj(f"{MOL_FILE_NAME[:-4]}_trj.xyz", initial_data, mean_func_coefs)
#model.update(dataset)
#model.optimize(dataset)

model.optimize(dataset)

model_chk = gpflow.utilities.deepcopy(model.model)
current_minima = tf.reduce_min(dataset.observations).numpy()

#print(f"Current dataset after init:\n{dataset}")

#left_borders, right_borders = roi_calc(model.model, MINIMA)

#This should be used!
rule = EfficientGlobalOptimization(ImprovementVariance(threshold=3))

#rule = EfficientGlobalOptimization(ExpectedImprovement())

#rule = EfficientGlobalOptimization(SparseGradExpectedImprovement(MINIMA, left_borders, right_borders))
#rule = EfficientGlobalOptimization(ExpectedImprovement())

#f = open("conf_search.log", "w+")

#print(dataset, file=f)

deepest_minima = []

early_termination_flag = False

for step in range(1, config.max_steps+1):
    print(f"Previous last_opt_ok: {LAST_OPT_OK}")
    print(f"Step number {step}")

    try:
        result = bo.optimize(1, dataset, model, rule, fit_initial_model=False)
        print(f"Optimization step {step} succeed!")
    except Exception:
        print("Optimization failed")
        print(result.astuple()[1][-1].dataset)
    
    print(f"After step: {LAST_OPT_OK}")

    last_opt_status = None
    with open(exp_name+"_last_opt_status.json", "r") as file:
        last_opt_status = json.load(file)
    print(last_opt_status)

    dataset = result.try_get_final_dataset()
    model = result.try_get_final_model()
    print(f"Last asked point was {ASKED_POINTS[-1]}")

    deepest_minima.append(tf.reduce_min(dataset.observations).numpy())    
    
    logs = {
        'acq_vals' : acq_vals_log,
        'deepest_minima' : deepest_minima
    }

    with open(f"{exp_name}_logs.json", 'w') as file:
        json.dump(logs, file)

    #print(f"Dataset size was {len(dataset)}")

    print(f"Eta is {rule._acquisition_function._eta}")    
    if LAST_OPT_OK:
        dataset = erase_last_from_dataset(dataset, 1)
        dataset = upd_dataset_from_trj(f"{MOL_FILE_NAME[:-4]}_trj.xyz", dataset)
    else:
        print(f"Last optimization finished with error, skipping trj parsing!")
    model.update(dataset)
    model.optimize(dataset)

    print("Updating model checkpoint!")
    model_chk = gpflow.utilities.deepcopy(model.model)
    current_minima = rule._acquisition_function._eta.numpy()[0]#tf.reduce_min(dataset.observations).numpy()
    print("Updated!")

    print(f"Step {step} complited! Current dataset is:\n{dataset}")
    
    if step < config.rolling_window_size:
        continue
    
    print(f"Checking termination criterion!")
    print(f"Acq vals in window: {logs['acq_vals'][max(0, step-config.rolling_window_size):step]}")
     
    rolling_mean = np.mean(logs['acq_vals'][max(0, step-config.rolling_window_size):step])
    rolling_std = np.std(logs['acq_vals'][max(0, step-config.rolling_window_size):step])

    print(f"After step {step}:")
    print(f"Current rolling mean of acqusition function maximum is: {rolling_mean}, threshold is {config.rolling_mean_threshold}")
    print(f"Current rolling std of acqusition function maximum is: {rolling_std}, threshold is {config.rolling_std_threshold}")
    if step >= config.rolling_window_size and rolling_std < config.rolling_std_threshold and rolling_mean < config.rolling_mean_threshold:
        print(f"Termination criterion reached on step {step}! Terminating search!")
        early_termination_flag = True
        break

    #left_borders, right_borders = roi_calc(model.model, MINIMA)
    
#    rule._acquisition_function.update_roi(
#        MINIMA,
#        left_borders,
#        right_borders,
#    )

    #print(f"Dataset size become {len(dataset)}")

    #print(model.model.parameters)

    #print(dataset.query_points.numpy(), file = f)

    #cur_prob = get_max_unknown_prob(model.model, dataset.query_points.numpy(), dataset.observations.numpy(), steps, mean_func_coefs)
    #save_plot(f"probs/plot_{steps}.html", dataset.query_points.numpy(), dataset.observations.numpy())
    #save_acq(f"probs/acq_{steps}.html", model.model, dataset.query_points.numpy(), dataset.observations.numpy())
    #save_prob(f"probs/prob_{steps}.html", model.model, dataset.query_points.numpy(), dataset.observations.numpy(), mean_func_coefs)
    #print(f"Current prob: {cur_prob}", file = f)
    
    #probs.append(cur_prob)

if not early_termination_flag:
    print("Max number of steps has been reached!")

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

dbscan_clf = DBSCAN(eps=np.pi/12,
                    min_samples=2,
                    metric=max_comp_dist).fit(query_points)

res = {int(label) : (1e9, -1) for label in np.unique(dbscan_clf.labels_)}

for i in range(len(query_points)):
    cluster_id = dbscan_clf.labels_[i]
    if observations[i] < res[cluster_id][0]:
        res[cluster_id] = observations[i].tolist(), i

print(f"Results of clustering: {res}\nThere are relative energy and number of structure for each cluster. Saved in `{exp_name}_clustering_results.json`")
json.dump(res, open(f'{exp_name}_clustering_results.json', 'w'))

print(f"Saving full dataset at `{exp_name}_all_points.json`")
json.dump(
    {
        'query_points' : query_points.tolist(),
        'observations' : observations.tolist()
    },
    open(f'{exp_name}_all_points.json', 'w')
)
