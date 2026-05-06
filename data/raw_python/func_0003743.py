def reciprocal(self):
        """The reciprocal of the unit cell

           In case of a three-dimensional periodic system, this is trivially the
           transpose of the inverse of the cell matrix. This means that each
           column of the matrix corresponds to a reciprocal cell vector. In case
           of lower-dimensional periodicity, the inactive columns are zero, and
           the active columns span the same sub space as the original cell
           vectors.
        """
        U, S, Vt = np.linalg.svd(self.matrix*self.active)
        Sinv = np.zeros(S.shape, float)
        for i in range(3):
            if abs(S[i]) < self.eps:
                Sinv[i] = 0.0
            else:
                Sinv[i] = 1.0/S[i]
        return np.dot(U*Sinv, Vt)*self.active