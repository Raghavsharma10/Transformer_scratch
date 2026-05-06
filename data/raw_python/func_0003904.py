def compare(self, other, r_threshold=1e-3):
        """Compare two rotations

           The RMSD of the rotation matrices is computed. The return value
           is True when the RMSD is below the threshold, i.e. when the two
           rotations are almost identical.
        """
        return compute_rmsd(self.r, other.r) < r_threshold