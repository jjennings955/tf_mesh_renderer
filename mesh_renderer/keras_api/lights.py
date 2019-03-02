import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

class Lights(object):
  def __init__(self, positions, intensities, _type='fixed', **kwargs):
    assert _type in ['fixed', 'variable']
    if isinstance(positions, np.ndarray):
        if _type == 'fixed':
            self.positions = K.constant(positions, name='light_positions')
        elif _type == 'variable':
            self.positions = K.variable(positions, name='light_positions')
        else:
            raise ValueError("type must be fixed or variable")
    elif isinstance(positions, (tf.Tensor, tf.Variable)):
        self.positions = positions
    else:
        raise ValueError("positions must be numpy array or tensor")

    if isinstance(intensities, np.ndarray):
        if _type == 'fixed':
            self.intensities = K.constant(positions, name='light_intensities')
        elif _type == 'variable':
            self.intensities = K.variable(positions, name='light_intensities')
        else:
            raise ValueError("type must be fixed or variable")
    elif isinstance(intensities, (tf.Tensor, tf.Variable)):
        self.intensities = intensities
    else:
        raise ValueError("positions must be numpy array or tensor")

    self.positions = positions
    self.intensities = intensities
    self._type = _type