def _next_sample_index(self):
        """StochasticMux chooses its next sample stream randomly"""
        return self.rng.choice(self.n_active,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))