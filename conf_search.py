#import calc # For init CURRENT_STRUCTURE_ID

from transform_kernel import TransformKernel

from coef_from_grid import pes, pes_tf

from calc import calc_energy, load_last_optimized_structure_xyz_block
from calc import change_dihedrals
from calc import parse_points_from_trj

from coef_calc import CoefCalculator

from sparse_ei import SparseExpectedImprovement
from explor_imp import ExplorationImprovement

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

from trieste.acquisition.function import ExpectedImprovement

from rdkit import Chem

from sklearn.cluster import DBSCAN

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

MOL_FILE_NAME = "tests/cur.mol"
NORM_ENERGY = 0.
RANDOM_DISPLACEMENT = True

DIHEDRAL_IDS = []

CUR_ADD_POINTS = []

global_degrees = []

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

def temp_calc(a : float, b  :float) -> float:
    """
        fast temp calc with cur dihedrals
    """
    if(tf.is_tensor(a)):
        a = a.numpy()
    if(tf.is_tensor(b)):
        b = b.numpy() 
    return (calc_energy(MOL_FILE_NAME, [([1, 2, 3, 4], a),\
                                         ([0, 1, 2, 3], b)], RANDOM_DISPLACEMENT) - NORM_ENERGY) * 627.509474063 

def calc(dihedrals : list[float]) -> float:
    """
        Perofrms calculating of energy with current dihedral angels
    """
    
    if tf.is_tensor(dihedrals):
        dihedrals = list(dihedrals.numpy())


    #Pre-opt
    print('Optimizing constrained struct')
    _ = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, RANDOM_DISPLACEMENT, constrained_opt=True)
    print('Optimized!\nLoading xyz from preopt')
    xyz_from_constrained = load_last_optimized_structure_xyz_block(MOL_FILE_NAME)
    print('Loaded!\nFull opt')
    en = calc_energy(MOL_FILE_NAME, list(zip(DIHEDRAL_IDS, dihedrals)), NORM_ENERGY, True, RANDOM_DISPLACEMENT, force_xyz_block=xyz_from_constrained)
    print('Optimized!')

    return en

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
    
    fig = go.Figure()
    xx = np.linspace(0, 2 * np.pi, 30)
    yy = xx
    xx_g, yy_g = np.meshgrid(xx, yy)
    pred_points = np.vstack((xx_g.flatten(), yy_g.flatten())).T

    cur_minimum = np.min(vals)

    points = np.array(points)
    vals = np.array(vals)

    mean, variance = model.predict_f(pred_points)
    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    acq_vals =  (cur_minimum - mean) * normal.cdf(cur_minimum) + variance * normal.prob(cur_minimum)
    rdists = tf.math.mod(tf.abs(tf.constant(pred_points.reshape(len(pred_points), 1, 2)) - tf.constant(points.reshape(1, len(points), 2))), 2 * np.pi)
    dists = tf.reshape(tf.reduce_min(
                   tf.math.reduce_max(tf.minimum(rdists, 2*np.pi - rdists), axis=-1), 
                                axis=-1), [pred_points.shape[0], 1])
    fig.add_surface(x=xx, y=yy, z=tf.where(dists > np.pi/6, acq_vals, 0).numpy().reshape((30, 30)))
    fig.write_html(file_name)

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
    
    print(f"Var: {var}\nMean: {mean}")

    print(f"Min var = {np.min(var)}, max var = {np.max(var)}")
    cur_minima = np.min(vals)
    prob = 0.5 * (erf((cur_minima + 3. - mean) / np.sqrt(2 * var)) + 1)
    
    max_unknown_prob = 0.
    plot_points = []
    plot_prob = []
    for i in range(len(mean)):
        plot_points.append(predicted_points[i])
        plot_prob.append(is_unique(predicted_points[i], points) * prob[i])
    fig.add_trace(go.Surface(x = xx, y = yy, z = np.array(plot_prob).reshape((30, 30)))) # plot_prob
    fig.add_trace(go.Surface(x = xx, y = yy, z = mean.reshape((30, 30)))) 
    qx, qy = zip(*points)
    fig.add_scatter3d(x = qx, y = qy, z = vals, mode = "markers")
    fig.write_html(file_name)

def is_unique(cur_point : list,
              points : list, 
              threshold : float = np.pi / 6) -> int:
    """
        Returns 1 if point stands alone in 'threshold'
        in every dimension, 0 otherway
    """  
    for point in points:
        had_seen = True
        for dim in range(len(cur_point)):
            had_seen = had_seen and (np.abs(cur_point[dim] - point[dim]) <= threshold)
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
    dataset : Dataset,
    coefs : list
) -> Dataset:
    """
        Return dataset that consists of old points
        add points from trj
    """
    
    degrees, energies = zip(*parse_points_from_trj(trj_filename, DIHEDRAL_IDS, NORM_ENERGY, True, 'structs/'))

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
    
    query_points = tf.slice(dataset.query_points, [0, 0], [dataset.query_points.shape[0] - n, 2])
    observations = tf.slice(dataset.observations, [0, 0], [dataset.observations.shape[0] - n, 1])

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

print("Coef calculator creatring")

coef_matrix = CoefCalculator(Chem.MolFromMolFile(MOL_FILE_NAME), "test_scans/").coef_matrix()

print("Good")

mean_func_coefs = []

for ids, coefs in coef_matrix:
    DIHEDRAL_IDS.append(ids)
    mean_func_coefs.append(coefs)

print(DIHEDRAL_IDS)
print(mean_func_coefs)

amps = np.array([
    np.abs(mean_func_coefs[i][:3]).sum() for i in range(len(mean_func_coefs))
])
print(amps)

potential_func = PotentialFunction(mean_func_coefs)

kernel = gpflow.kernels.White(0.001) + gpflow.kernels.Periodic(gpflow.kernels.Matern12(variance=0.3, lengthscales=0.005, active_dims=[0,1]), period=[2*np.pi, 2*np.pi]) + TransformKernel(potential_func, gpflow.kernels.Matern12(variance=0.3, lengthscales=0.005, active_dims=[0, 1]))

print(kernel)

search_space = Box([0., 0.], [2 * np.pi, 2 * np.pi])  # define the search space directly

#Calc normalizing energy
#in kcal/mol!

NORM_ENERGY = -367398.19960427243

print(NORM_ENERGY)

observer = trieste.objectives.utils.mk_observer(func) # defines a observer of our 'func'

# calculating initial points
num_initial_points = 1
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

gpr = gpflow.models.GPR(
    initial_data.astuple(), 
    kernel
)

print(gpr.parameters)
gpflow.set_trainable(gpr.likelihood, False)
gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)

model = GaussianProcessRegression(gpr, num_kernel_samples=100)

# Starting Bayesian optimization
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

print(f"Initital data:\n{initial_data}")

dataset = upd_dataset_from_trj("tests/cur_trj.xyz", None, mean_func_coefs)
model.update(dataset)
model.optimize(dataset)

print(f"Current dataset after init:\n{dataset}")

rule = EfficientGlobalOptimization(SparseExpectedImprovement(threshold=np.pi/6))

prev_result = None

f = open("conf_search.log", "w+")

print(dataset, file=f)

probs = []

steps = 1
print("Begin opt!", file = f)
for _ in range(50):
    print(f"Step number {steps}")
    try:
        result = bo.optimize(1, dataset, model, rule, fit_initial_model = False)
        print(f"Optimization step {steps} succeed!", file = f)
    except Exception:
        print("Optimization failed", file = f)
        print(result.astuple()[1][-1].dataset, file = f)
    dataset = result.try_get_final_dataset()
    model = result.try_get_final_model()

    print(dataset)
    
    print(f"Dataset size was {len(dataset)}")
    
    dataset = erase_last_from_dataset(dataset, 1)
    dataset = upd_dataset_from_trj("tests/cur_trj.xyz", dataset, mean_func_coefs)
    model.update(dataset)
    model.optimize(dataset)

    print(f"Dataset size become {len(dataset)}")

    print(model.model.parameters)

    print(dataset.query_points.numpy(), file = f)

    cur_prob = get_max_unknown_prob(model.model, dataset.query_points.numpy(), dataset.observations.numpy(), steps, mean_func_coefs)
    save_plot(f"probs/plot_{steps}.html", dataset.query_points.numpy(), dataset.observations.numpy())
    save_acq(f"probs/acq_{steps}.html", model.model, dataset.query_points.numpy(), dataset.observations.numpy())

    print(f"Current prob: {cur_prob}", file = f)
    steps += 1
    prev_result = result
    
    probs.append(cur_prob)

# printing results
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}", file = f)
print(f"observation: {observations[arg_min_idx, :]}", file = f)
save_res("tests/res.xyz", *query_points[arg_min_idx, :])
save_all("tests/all.xyz", query_points)

print(probs, file=f)

f.close()

dbscan_clf = DBSCAN(eps=np.pi/12,
                    min_samples=2,
                    metric=max_comp_dist).fit(query_points)

res = {label : (1e9, -1) for label in np.unique(dbscan_clf.labels_)}

for i in range(len(query_points)):
    cluster_id = dbscan_clf.labels_[i]
    if observations[i] < res[cluster_id][0]:
        res[cluster_id] = observations[i], i

print(res)

with open("res.dat", "w+") as file:
    file.write(query_points.__str__())
    file.write("\n")
    file.write(observations.__str__())
save_plot("tests/plot.html", query_points, observations)
save_prob("tests/prob.html", result.try_get_final_model().model, query_points, observations)
