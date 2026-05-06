def sub_hmm(self, states):
        r""" Returns HMM on a subset of states

        Returns the HMM restricted to the selected subset of states.
        Will raise exception if the hidden transition matrix cannot be normalized on this subset

        """
        # restrict initial distribution
        pi_sub = self._Pi[states]
        pi_sub /= pi_sub.sum()

        # restrict transition matrix
        P_sub = self._Tij[states, :][:, states]
        # checks if this selection is possible
        assert np.all(P_sub.sum(axis=1) > 0), \
            'Illegal sub_hmm request: transition matrix cannot be normalized on ' + str(states)
        P_sub /= P_sub.sum(axis=1)[:, None]

        # restrict output model
        out_sub = self.output_model.sub_output_model(states)

        return HMM(pi_sub, P_sub, out_sub, lag=self.lag)