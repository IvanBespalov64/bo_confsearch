import numpy as np
import tensorflow as tf

from trieste.types import TensorType
from trieste.data import Dataset

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import ExpectedImprovement
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder)
from trieste.models import ProbabilisticModel

from typing import cast

class ExplorationImprovement(SingleModelAcquisitionBuilder):
    """
        Acquisition function that is a probability 
        to find minima lower than we already had for 
        self._threshold. Acquisiton function has a 
        zero-value in points, that are closer then dist
        in each dimension
    """
    def __init__(self, threshold, dist):
        """
            initiates threshold
        """
        self._threshold = threshold
        self._dist = dist
    
    def __repr__(self) -> str:
        """"""
        return "SparseExpectedImprovement()"
    
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return exploration_improvement(model, eta, dataset, self._threshold, self._dist)
    
    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Dataset = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, exploration_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta, dataset, self._threshold, self._dist)  # type: ignore
        return function
    
    
class exploration_improvement(AcquisitionFunctionClass):
    def __init__(self, 
                 model: ProbabilisticModel, 
                 eta: TensorType, 
                 dataset : Dataset, 
                 threshold : float, 
                 dist : float):
        """"""
        self._model = model
        self._eta = tf.Variable(eta)
        self._dataset = dataset
        self._threshold = threshold
        self._dist = dist

    def update(self, 
               eta: TensorType, 
               dataset : Dataset, 
               threshold : float,
               dist : float) -> None:
        """Update the acquisition function with a new eta value, dataset, threshold and dist"""
        self._eta.assign(eta)
        self._dataset = dataset
        self._threshold = threshold
        self._dist = dist

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
#         tf.debugging.assert_shapes(
#             [(x, [..., 1, None])],
#             message="This acquisition function only supports batch sizes of one.",
#         )
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        rdists = tf.math.mod(tf.abs(x - self._dataset.query_points), 2 * np.pi)
        # all torsion angles should be less then self._dist
        dists = tf.reshape(tf.reduce_min(tf.math.reduce_max(tf.minimum(rdists, 2*np.pi - rdists), axis=-1), axis=-1), [x.shape[0], 1])
        
        prob = 0.5 * (tf.math.erf((self._eta + self._threshold - mean) / tf.math.sqrt(2 * variance)) + 1)
        
        return tf.where(dists > self._dist, prob, 0) #0.01 * prob except 0?
    
#rule = EfficientGlobalOptimization(ExplorationImprovement(threshold=3., dist=np.pi / 12))
