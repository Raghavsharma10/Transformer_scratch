def _next_sample_index(self):
        """ShuffledMux chooses its next sample stream randomly,
        conditioned on the stream weights.
        """
        return self.rng.choice(self.n_streams,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))