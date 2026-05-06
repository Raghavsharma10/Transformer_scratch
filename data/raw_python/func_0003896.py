def compare(self, other, t_threshold=1e-3):
        """Compare two translations

           The RMSD of the translation vectors is computed. The return value
           is True when the RMSD is below the threshold, i.e. when the two
           translations are almost identical.
        """
        return compute_rmsd(self.t, other.t) < t_threshold