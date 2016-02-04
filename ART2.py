"""
# ART2
Adaptive Resonance Theory Neural Networks
by Aman Ahuja | github.com/amanahuja | twitter: @amanqa

## Notes
1. ART that accommodates patterns with continuous valued components. Since
continuous input patterns may be arbitrarily close together, ART2
introduces normalization and noise suppression.

2. ART2 treats small components as noise and does not distinguish between
patterns that are merely scaled versions of each other.

## F1 layer
F1 layers consists of (six) types of units :
  - W, X, U, V, P, and Q
  - There are n units of each type: ($6n$ units)
  - There are 3 supplemental units, one each between
        W and X units, P and Q units, and V and U units. Designated
        WN, VN, and PN, respectively.
  - Thus, there are 6n + 3 units in the F1 layer.

Unit roles:
  - $U$ are analagous to input phase of ART1's input layer F1 [(F0? -AA)]
  - Units X and Q inhibit any vector components that fall below
    the user-seleted parameter theta, for noise suppression.

## F2 layer
The F2 competition layer forms the network output. F2 nodes are called cluster
units.
  - F2 nodes are designated Y, same as in ART1
  - Cluster Unit compete to learn each input pattern. Candidacy and selection
    are similar to ART1.
  - As ART1, learning occurs IFF top-down weight vector for the candidate
    unit J is sufficiently similar to input vector, as determined by
    the vigilance parameter rho.
  - The reset test for ART2 is different from ART1
  - Activation of the winning/candidate F2 unit is designated D

## Resonance Learning
Learning consists of the adjustment of weights between F1 and F2 units.

  - Unlike the ART1 case, which deals with only binary values, the differential
    equations for learning do not simplify. Instead the weights for ART1
    reach equilibrium through iteration.
  - In "fast learning" mode, a single input pattern is used to adjust weights
    until they achieve stability.

Where:
 - n is len(input pattern)
"""

import numpy as np


class ART2(object):

    def __init__(self, n=5, m=3, rho=0.9, theta=None):
        """
        Create ART2 network with specified shape

        For Input array I of size n, we need n input nodes in F1.


        Parameters:
        -----------
        n : int
            feature dimension of input; number of nodes in F1
        m : int
            Number of neurons in F2 competition layer
            max number of categories
            compare to n_class
        rho : float
            Vigilance parameter
            larger rho: less inclusive prototypes
            smaller rho: more generalization
        theta :
            Suppression paramater
        L : float
            Learning parameter: # TODO

        internal parameters
        ----------
        Bij: array of shape (m x n)
            Feed-Forward weights
        Tji: array of shape (n x m)
            Feed-back weights
        """

        self.input_size = n
        self.output_size = m

        """init layers
        F0 --> F1 --> F2
        S  --> X  --> Y
        """
        # F2
        self.yj = np.zeros(self.output_size)
        self.active_cluster_units = []
        # F1
        self.si = np.zeros(self.input_size)
        # F0
        self.si = np.zeros(self.input_size)

        """init parameters"""
        self.params = {}
        # a,b fixed weights in F1 layer; should not be zero
        self.params['a'] = 10
        self.params['b'] = 10
        # c fixed weight used in testing for reset
        self.params['c'] = 0.1
        # d activation of winning F2 unit
        self.params['d'] = 0.9
        # c*d / (1-d)  must be less than or equal to one
        # as ratio --> 1 for greater vigilance
        self.params['e'] = 0.00001
        # small param to prevent division by zero

        # self.L = 2
        # rho : vigilance parameter
        self.rho = rho
        # theta: noise suppression parameter
        #   e.g. theta = 1 / sqrt(n)
        if theta is None:
            self.theta = 1 / np.sqrt(self.input_size)
        else:
            self.theta = theta
        # alpha: learning rate. Small value : slower learning,
        #  but also more likely to reach equilibrium in slow
        # learning mode
        self.alpha = 0.6

        """init weights"""
        # Bij initially (7.0, 7.0) for each cluster unit
        self.Bij = np.ones((n, m)) * 7.0
        # Tji initially 0
        self.Tji = np.zeros((m, n))

        """init other activations"""
        self.ui = None
        self.vi = None
