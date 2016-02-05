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
from numpy.linalg import norm

# Logging conf
import logging
import sys


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
        self.xi = np.zeros(self.input_size)
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
        self.Bij = np.ones((n, m)) * 5.0
        # Tji initially 0
        self.Tji = np.zeros((m, n))

        """init other activations"""
        self.ui = np.zeros(self.input_size)
        self.vi = None

        """Other helpers"""
        self.log = None

    def compute(self, all_data):
        """Process and learn from all data
        Step 1
        fast learning: repeat this step until placement of
        patterns on cluster units does not change
        from one epoch to the next
        """
        for iepoch in range(self.n_epochs):
            self._training_epoch(all_data)

            # test stopping condition for n_epochs

        return True

    def learning_trial(self, idata):
        """
        Step 3-11
        idata is a single row of input
        A learning trial consists of one presentation of one input pattern.
        V and P will reach equilibrium after two updates of F1
        """

        self.log.info("Starting Learning Trial.")
        self.log.debug("input pattern: {}".format(idata))
        self.log.debug("theta: {}".format(self.theta))

        # at beginning of learning trial, set all activations to zero
        self._zero_activations()

        self.si = idata
        # TODO: Should this be here?

        # Update F1 activations, no candidate cluster unit
        self._update_F1_activation()

        # Update F1 activations again
        self._update_F1_activation()

        """
        After F1 activations achieve equilibrium
        TMP: Assume only two F1 updates needed for now
        Then proceed feed-forward to F2
        """
        # TODO: instead check if ui or pi will change significantly

        # now P units send signals to F2 layer
        self.yj = np.dot(self.Bij.T, self.pi)

        J = self._select_candidate_cluster_unit()

        """step 8 (resonance)
        reset cannot occur during resonance
        new winning unit (J) cannot be chosen during resonance

        """
        if len(self.active_cluster_units) == 0:
            self._update_weights_first_pattern(J)
        else:
            self._resonance_learning(J)

        # add J to active list
        if J not in self.active_cluster_units:
            self.active_cluster_units.append(J)

        return True

    def _training_epoch(self, all_data):

        # initialize parameters and weights
        pass  # done in __init__

        for idata in all_data:
            self.si = idata  # input vector F0

            self.learning_trial()

        return True

    def _select_candidate_cluster_unit(self):
        """ RESET LOOP
        This loop selects an appropriate candidate cluster unit for learninig
         - Each iteration selects a candidate unit.
         - Iterations continue until reset condition is met (reset is False)
         - if a candidate unit does not satisfy, it is inhibited and can not be
         selected again in this presentation of the input pattern.

        No learning occurs in this phase.

        returns:
            J, the index of the selected cluster unit
        """
        self.reset = True
        while self.reset:
            self.log.info("candidate selection loop iter start")
            #  check reset

            # Select best active candidate
            # ... largest element of Y that is not inhibited
            J = np.argmax(self.yj)  # J current candidate, not same as index jj

            self.log.debug("\tyj: {}".format(self.yj))
            self.log.debug("\tpicking J = {}".format(J))
            # Test stopping condition here
            # (check reset)

            e = self.params['e']

            #  confirm candidate: inhibit or proceed
            if (self.vi == 0).all():
                self.ui = np.zeros(self.input_size)
            else:
                self.ui = self.vi / (e + norm(self.vi))
            # pi =

            # calculate ri (reset node)
            c = self.params['c']
            term1 = norm(self.ui + c*self.ui)
            term2 = norm(self.ui) + c*norm(self.ui)
            self.ri = term1 / term2

            if self.ri >= (self.rho - e):
                self.log.info("\tReset is False: Candidate is good.")
                # Reset condition satisfied: cluster unit may learn
                self.reset = False

                # finish updating F1 activations
                self._update_F1_activation()
                # TODO: this will update ui twice. Confirm ok
            elif self.ri < (self.rho - e):
                self.reset = True
                self.log.info("\treset is True")
                self.yj[J] = -1.0

            # break inf loop manually
            # self.log.warn("EXIT RESET LOOP MANUALLY")
            # self.reset = False

        return J

    def _resonance_learning(self, J, n_iter=20):
        """
        Learn on confirmed candidate
        In slow learning, only one update of weights in this trial
            n_learning_iterations = 1
            we then present the next input pattern

        In fast learning, present input again (same learning trial)
          - until weights reach equilibrium for this trial
          - presentation is: "weight-update-F1-update"
        """
        self.log.info("Entering Resonance phase with J = {}".format(J))

        for ilearn in range(n_iter):
            self.log.info("learning iter start")

            self._update_weights(J)

            # in slow learning, this step not required?
            D = np.ones(self.output_size)
            self._update_F1_activation(J, D)

            # test stopping condition for weight updates
            # if change in weights was below some tolerance

        return True

    def _update_weights_first_pattern(self, J):
        """Equilibrium weights for the first pattern presented
        converge to these values. This shortcut can save many
        iterations.
        """
        self.log.info("Weight update using first-pattern shortcut")
        # Fast learning first pattern simplification
        d = self.params['d']
        self.Tji[J, :] = self.ui / (1 - d)
        self.Bij[:, J] = self.ui / (1 - d)

        # log
        self.log.debug("Tji[J]: {}".format(self.Tji[J, :]))
        self.log.debug("Bij[J]: {}".format(self.Bij[:, J]))

        return

    def _update_weights(self, J):
        """update weights
        for Tji and Bij
        """
        self.log.info("Updating Weights")

        # get useful terms
        alpha = self.alpha
        d = self.params['d']

        term1 = alpha*d*self.ui
        term2 = (1 + alpha*d*(d - 1))

        self.Tji[J, :] = term1 + term2*self.Tji[J, :]
        self.Bij[:, J] = term1 + term2*self.Bij[:, J]

        # log
        self.log.debug("Tji[J]: {}".format(self.Tji[J, :]))
        self.log.debug("Bij[J]: {}".format(self.Bij[:, J]))

        return

    def _update_F1_activation(self, J=None, D=None):
        """
        if winning unit has been selected
          J is winning cluster unit
          D is F2 activation
        else if no winning unit selected
          J is None
          D is zero vector

        """
        # Checks
        # self.log.warn("Warning: Skipping J xor D check!")
        # if (J is None) ^ (D is None):
        #     raise Exception("Must provide both J and D, or neither.")

        msg = "Updating F1 activations"
        if J is not None:
            msg = msg + " with J = {}".format(J)
        self.log.info(msg)

        a = self.params['a']
        b = self.params['b']
        d = self.params['d']
        e = self.params['e']

        # compute activation of Unit Ui
        #  - activation of Vi normalized to unit length
        if self.vi is None:
            self.ui = np.zeros(self.input_size)
        else:
            self.ui = self.vi / (e + norm(self.vi))

        # signal sent from each unit Ui to associated Wi and Pi

        # compute activation of Wi

        self.wi = self.si + a * self.ui
        # compute activation of pi
        # WRONG: self.pi = self.ui + np.dot(self.yj, self.Tji)
        if J is not None:
            self.pi = self.ui + d * self.Tji[J, :]
        else:
            self.pi = self.ui

        # TODO: consider RESET here

        # compute activation of Xi
        # self.xi = self._thresh(self.wi / norm(self.wi))
        self.xi = self.wi / (e + norm(self.wi))
        # compute activation of Qi
        # self.qi = self._thresh(self.pi / (e + norm(self.pi)))
        self.qi = self.pi / (e + norm(self.pi))

        # send signal to Vi
        self.vi = self._thresh(self.xi) + b * self._thresh(self.qi)

        self._log_values()
        return True

    """Helper methods"""
    def _zero_activations(self):
        """Set activations to zero
        common operation, e.g. beginning of a learning trial
        """
        self.log.debug("zero'ing activations")
        self.si = np.zeros(self.input_size)
        self.ui = np.zeros(self.input_size)
        self.vi = np.zeros(self.input_size)
        return

    def _thresh(self, vec):
        """
        This function treats any signal that is less than theta
        as noise and suppresses it (sets it to zero). The value
        of the parameter theta is specified by the user.
        """
        assert isinstance(vec, np.ndarray), "type check"
        cpy = vec.copy()
        cpy[cpy < self.theta] = 0
        return cpy

    def _clean_input_pattern(self, idata):
        assert len(idata) == self.input_size, "size check"
        assert isinstance(idata, np.ndarray), "type check"

        return idata

    """Logging Functions"""
    def stop_logging(self):
        """Logging stuff
        closes filehandlers and stuff
        """
        self.log.info('Stop Logging.')
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.close()
            self.log.removeHandler(handler)
        self.log = None

    def start_logging(self, to_file=True, to_console=True):
        """Logging!
        init logging handlers and stuff
        to_file and to_console are booleans
        # TODO: accept logging level
        """
        # remove any existing logger
        if self.log is not None:
            self.stop_logging()
            self.log = None

        # Create logger and configure
        self.log = logging.getLogger('ann.art.art2')
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        formatter = logging.Formatter(
            fmt='%(levelname)8s:%(message)s'
        )

        # add file logging
        if to_file:
            fh = logging.FileHandler(
                filename='ART_LOG.log',
                mode='w',
            )
            fh.setFormatter(formatter)
            fh.setLevel(logging.WARN)
            self.log.addHandler(fh)

        # create console handler with a lower log level for debugging
        if to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(logging.DEBUG)
            self.log.addHandler(ch)

        self.log.info('Start Logging')

    def getlogger(self):
        """Logging stuff
        """
        return self.log

    def _log_values(self, J=None):
        """Logging stuff
        convenience function
        """
        self.log.debug("\t--- debug values --- ")
        self.log.debug("\tui : {}".format(self.ui))
        self.log.debug("\twi : {}".format(self.wi))
        self.log.debug("\tpi : {}".format(self.pi))
        self.log.debug("\txi : {}".format(self.xi))
        self.log.debug("\tqi : {}".format(self.qi))
        self.log.debug("\tvi : {}".format(self.vi))
        if J is not None:
            self.log.debug("\tWeights with J = {}".format(J))
            self.log.debug("\tBij: {}".format(self.bij[:, J]))
            self.log.debug("\tTji: {}".format(self.tji[J, :]))
