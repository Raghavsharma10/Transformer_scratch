def set_transition_down(self, p_self):
        '''Set the downbeat-tracking transition matrix according to
        self-loop probabilities.

        Parameters
        ----------
        p_self : None, float in (0, 1), or np.ndarray [shape=(2,)]
            Optional self-loop probability(ies), used for Viterbi decoding
        '''
        if p_self is None:
            self.down_transition = None
        else:
            self.down_transition = transition_loop(2, p_self)