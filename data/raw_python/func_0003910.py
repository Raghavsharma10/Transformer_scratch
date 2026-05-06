def compare(self, other, t_threshold=1e-3, r_threshold=1e-3):
        """Compare two transformations

           The RMSD values of the rotation matrices and the translation vectors
           are computed. The return value is True when the RMSD values are below
           the thresholds, i.e. when the two transformations are almost
           identical.
        """
        return compute_rmsd(self.t, other.t) < t_threshold and compute_rmsd(self.r, other.r) < r_threshold