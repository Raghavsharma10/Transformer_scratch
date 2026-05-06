def set_transition(self, p_self):
        '''Set the transition matrix according to self-loop probabilities.

        Parameters
        ----------
        p_self : None, float in (0, 1), or np.ndarray [shape=(n_labels,)]
            Optional self-loop probability(ies), used for Viterbi decoding
        '''
        if p_self is None:
            self.transition = None
        else:
            self.transition = transition_loop(len(self._classes), p_self)