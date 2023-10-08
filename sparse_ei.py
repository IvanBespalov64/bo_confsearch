import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.types import TensorType
from trieste.data import Dataset

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import ExpectedImprovement
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass)
from trieste.models import ProbabilisticModel

from typing import cast

class SparseExpectedImprovement(ExpectedImprovement):
    """
        Realizes EI acquisition function but with the criteria,
        that the points, that are closer then <threshold>, are
        not interesting (Acquisition function equals 0). Distance 
        is the l1-norm
    """
    def __init__(self, threshold):
        """
            initiates threshold
        """
        self._threshold = threshold
    
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
        
        return sparse_expected_improvement(model, eta, dataset, self._threshold)
    
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
        tf.debugging.Assert(isinstance(function, sparse_expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta, dataset, self._threshold)  # type: ignore
        return function
    
    
class sparse_expected_improvement(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, eta: TensorType, dataset : Dataset, threshold : float):
        """"""
        self._model = model
        self._eta = tf.Variable(eta)
        self._dataset = dataset
        self._threshold = threshold

    def update(self, eta: TensorType, dataset : Dataset, threshold : float) -> None:
        """Update the acquisition function with a new eta value, dataset and threshold."""
        self._eta.assign(eta)
        self._dataset = dataset
        self._threshold = threshold

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
#         tf.debugging.assert_shapes(
#             [(x, [..., 1, None])],
#             message="This acquisition function only supports batch sizes of one.",
#         )
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        rdists = tf.math.mod(tf.abs(x - self._dataset.query_points), 2 * np.pi)
        # all torsion angles should be less then self._threshold
        dists = tf.reshape(
                tf.reduce_min(
                    tf.math.reduce_max(
                        tf.minimum(rdists, 2*np.pi - rdists), 
                        axis=-1), 
                    axis=-1), 
                [x.shape[0], 1]
            )
        acq_vals = (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta)
        return tf.where(dists > self._threshold, acq_vals, 0) 
