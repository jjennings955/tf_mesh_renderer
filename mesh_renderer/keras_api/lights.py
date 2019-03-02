import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

class Lights(object):
  def __init__(self, positions, intensities):
    self.positions = positions
    self.intensities = intensities