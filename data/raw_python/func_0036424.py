def _similarity_matrix(self, concepts):
        """
        Computes a semantic similarity matrix for a set of concepts.
        """
        n_cons = len(concepts)
        sim_mat = np.zeros((n_cons, n_cons))
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                # Just build the lower triangle
                if i >= j:
                    sim_mat[i,j] = self._semsim(c1, c2) if i != j else 1.
        return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())