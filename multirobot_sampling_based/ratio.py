# %%
########################################################################
# This files holds some helper functions to choose leg_lengths or
# velocities.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
from itertools import permutations
import numpy as np

np.random.seed(42)

if os.name == "posix":
    os.system("clear")


########## Classes #####################################################
class Ratio:
    def __init__(self, n_robot, n_mode, vels, repeatable=True, tumbling=False):
        self._n_robot = n_robot
        self._n_mode = n_mode
        self._vels = vels
        self._tumbling = tumbling
        if repeatable:
            self._rnd = np.random.default_rng(42)
        else:
            self._rnd = np.random.default_rng()
        self._set_combos()
        self._set_matrices()

    @property
    def matrices(self):
        return self._matrices

    def _set_combos(self):
        # If vels is shorter than number of robots, repeat it.
        seed = np.unique(self._vels).tolist()
        if len(seed) < self._n_robot:
            n_rep = int(self._n_robot // len(seed)) + 1
        else:
            n_rep = 2
        seed = seed * n_rep
        #
        if self._n_mode < 6:
            combos = list(set(permutations(seed, self._n_robot)))
        else:
            combos = self._rnd.choice(self._vels, (1000, self._n_robot))
        self._combos = combos

    def _set_matrices(self):
        matrices = []
        for combo in self._combos:
            B = np.zeros((self._n_mode, self._n_robot))
            if self._tumbling:
                B[0] = 1.0
                iterator = range(1, self._n_mode)
            else:
                iterator = range(0, self._n_mode)
            # Make aggrergeated B.
            for i in iterator:
                B[i] = np.roll(combo, -i)
            B = B.T
            # Make beta matrix
            beta = B / B[0]
            # If it is controllable add it
            if np.linalg.matrix_rank(beta) >= self._n_robot:
                _, e, _ = np.linalg.svd(beta, full_matrices=False)
                # e = e/e[0]
                det = e.prod()
                matrices.append((B, beta, e, det))
        # Sort matrices based on ascending determinent and eigen values.
        matrices = sorted(
            matrices, key=lambda x: (abs(x[3]), *sorted(abs(x[2])))
        )
        self._matrices = matrices

    def print(self):
        for B, beta, e, det in self._matrices:
            msg = []
            # Format beta and B rows
            for row_beta, row_B in zip(beta, B):
                beta_str = ", ".join(f"{x:05.2f}" for x in row_beta)
                B_str = ", ".join(f"{x:05.2f}" for x in row_B)
                msg.append(f"{beta_str} | {B_str}")
            # Print det and e
            det_str = f"det: {det:+010.6f}"
            e_str = ", ".join(f"{x:05.2f}" for x in e)
            msg.append(f"{det_str} | {e_str}")
            print("\n".join(msg))
            print("*" * 79)

    def get_best(self):
        # Return best B
        if len(self._matrices):
            return self._matrices[-1][0], self._matrices[-1][2]
        else:
            return None


def test():
    vels = [2, 1]
    n_mode = 10
    n_robot = 10
    ratios = Ratio(n_robot, n_mode, vels, tumbling=False)
    ratios.print()
    print(ratios.get_best())


########## test section ################################################
if __name__ == "__main__":
    test()
