def set_transition_beat(self, p_self):
        '''Set the beat-tracking transition matrix according to
        self-loop probabilities.

        Parameters
        ----------
        p_self : None, float in (0, 1), or np.ndarray [shape=(2,)]
            Optional self-loop probability(ies), used for Viterbi decoding
        '''
        if p_self is None:
            self.beat_transition = None
        else:
            self.beat_transition = transition_loop(2, p_self)