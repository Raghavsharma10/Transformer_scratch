def lifetimes(self):
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(nstates)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-tau / ln | p_{ii} |, i = 1,...,nstates`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.

        """
        return -self._lag / np.log(np.diag(self.transition_matrix))