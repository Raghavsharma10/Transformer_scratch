def solve(self, angles):
        """Calculate a position of the end-effector and return it."""
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self._matrices(angles)),
            np.array([0., 0., 0., 1.])
        )[:3]