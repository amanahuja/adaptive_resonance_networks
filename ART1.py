"""
ART1
Adaptive Resonance Theory Neural Networks
by Aman Ahuja | github.com/amanahuja | twitter: @amanqa
"""

import numpy as np


class ART1:
    """
    ART1 networks process binary inputs only.

    Usage example:
    --------------
    # Create a ART network with input of size 5 and 10 cluster (output) units
    >>> network = ART(n=5, m=10, rho=0.5)
    """

    def __init__(self, n=5, m=10, rho=.5):
        """
        Create network with specified shape

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
        L : float
            Learning parameter: # TODO

        internal parameters
        ----------
        S: array of size (n)
            F0 units activation vector
        X: array of size (n)
            F1 units activation vector
        Y: array of size (m)
            F2 units activation vector
        Bij: array of shape (m x n)
            Feed-Forward weights
        Tji: array of shape (n x m)
            Feed-back weights
        n_cats : int
            Number of F2 neurons that are active
            (at any given time, number of category templates)

        """
        self.input_size = n
        self.output_size = m

        """
        Intitialize layers
        # F0 --> F1 --> F2
        # S  --> X  --> Y
        """
        # Input Layer: F0
        self.S = np.zeros(self.input_size)

        # Comparison layer: F1
        self.X = np.zeros(self.input_size)

        # Recognition layer: F2
        self.Y = np.zeros(self.output_size)

        """init paramaters"""
        # L learning paramater
        self.L = 2
        # Vigilance parameter
        self.rho = rho
        # Number of active units in F2
        self.n_cats = 0
        # number of epochs
        # for ART1, just use n_epochs = 1
        self.n_epochs = 1

        """init weights"""
        # Feed-forward weights
        # The initial bottom-up weights should be smaller than or equal to the
        # equilibrium value. Generally, larger initial bottom-up weights favor
        # creation of new nodes over attempting to put a pattern on a
        # previously trained cluster unit.
        self._initial_bottomup_weights = 0.2
        self.Bij = np.ones((n, m)) * self._initial_bottomup_weights

        # Feed-back weights
        # for ART1 set these to 1, ensures reset node doesn't reject an
        # active but uncommitted cluster unit.
        self.Tji = np.ones((m, n)) * 1

    def clear(self):
        """Reset whole network to start conditions
        """
        n = self.input_size
        m = self.output_size
        self.n_cats = 0
        self.S = np.zeros(self.input_size)
        self.X = np.zeros(self.input_size)
        self.Y = np.zeros(self.output_size)
        self.Bij = np.ones((n, m)) * self._initial_bottomup_weights
        self.Tji = np.ones((m, n)) * 1

    def compute(self, all_data):
        """Process and learn from all data
        """
        for iepoch in range(self.n_epochs):
            self.training_epoch(all_data)

        return True

    def training_epoch(self, all_data):
        """single epoch of training
        resonance may occur (multiple passes)
        processes all the data
        """
        # initialize paramaters and weights
        pass  # TODO: self.reset

        for idata in all_data:
            idata = self.sanitize_input(idata)

            self.present_input_pattern(idata)

        return True

    def sanitize_input(self, idata):
        """placeholder for helper function
        """
        assert len(idata) == self.input_size, "size check"
        assert isinstance(idata, np.ndarray), "type check"
        return idata

    def present_input_pattern(self, idata):
        """
        idata is a single row of input
        """
        # Set activation of all F2 units to zero
        self.Y = np.zeros(self.output_size)

        # set activation of F0 to input vector
        self.S = idata

        # norm of S
        S_norm = self.S.sum()

        # activation of F1
        # Q: Shouldn't activation be calc'ed with weights?
        self.X = self.S

        # feed forward to F2
        self.Y = np.dot(self.Bij.T, idata)

        self.reset = True
        while self.reset:
            # Select best active candidate
            # ... largest element of Y that is not inhibited

            J = np.argmax(self.Y)   # J current candidate, not same as index jj

            # Test stopping condition here
            if np.all(self.Y == -1):
                self.log.warn("Stopping condition raised.")
                self.log.warn("-" * 20)
                self.log.warn("Input ({}) cannot be classified.".format(
                    idata))
                self.log.warn("Possible outlier / Decrease vigilance.")
                break

            # recompute F1 activation (yes again)
            # xi = si*tji with j = J

            si = self.S
            ti = self.Tji[J, :]
            self.X = si*ti

            # compute norm of X
            X_norm = self.X.sum()

            # reset test
            match = X_norm / S_norm
            if match >= self.rho:
                self.reset = False
            else:
                self.reset = True
                self.Y[J] = -1.0

        # Done with while loop

        # Step 12: update weights

        # update Bij weights (for j = J = 1)
        xi = self.X[:]
        bi_new = self.L*xi / (self.L - 1 + X_norm)
        self.Bij[:, J] = bi_new

        # update Tji weights (for j = J = 1)
        self.Tji[J, :] = xi

        return

    def predict(self, X):
        """helper func
        change no weights, find most active cluster unit
        for a given input pattern.
        X should be a single row of input (idata)
        """
        C = np.dot(self.Bij.T, X)

        # check if no cluster units are active!
        # TODO

        return np.argmax(C)

    def _OLD_validate_data(self, dat):
        """
        dat is a single input record
        Checks: data must be 1s and 0s
        """
        pass_checks = True

        # Dimensions must match
        if dat.shape[0] != len(self.F1):
            pass_checks = False
            msg = "Input dimensins mismatch."

        # Data must be 1s or 0s
        if not np.all((dat == 1) | (dat == 0)):
            pass_checks = False
            msg = "Input must be binary."

        if pass_checks:
            return True
        else:
            raise Exception("Data does not validate: {}".format(msg))
