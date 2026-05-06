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
            self.transition = np.empty((len(self._classes), 2, 2))
            if np.isscalar(p_self):
                self.transition = transition_loop(2, p_self)
            elif len(p_self) != len(self._classes):
                raise ParameterError('Invalid p_self.shape={} for vocabulary size={}'.format(p_self.shape, len(self._classes)))
            else:
                for i in range(len(self._classes)):
                    self.transition[i] = transition_loop(2, p_self[i])