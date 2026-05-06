def _check_r(self, r):
        """the columns must orthogonal"""
        if abs(np.dot(r[:, 0], r[:, 0]) - 1) > eps or \
            abs(np.dot(r[:, 0], r[:, 0]) - 1) > eps or \
            abs(np.dot(r[:, 0], r[:, 0]) - 1) > eps or \
            np.dot(r[:, 0], r[:, 1]) > eps or \
            np.dot(r[:, 1], r[:, 2]) > eps or \
            np.dot(r[:, 2], r[:, 0]) > eps:
            raise ValueError("The rotation matrix is significantly non-orthonormal.")