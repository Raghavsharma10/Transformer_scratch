def _activate_stream(self, idx):
        '''Randomly select and create a stream.

        StochasticMux adds mode handling to _activate_stream, making it so that
        if we're not sampling "with_replacement", the distribution for this
        chosen streamer is set to 0, causing the streamer not to be available
        until it is exhausted.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Get the number of samples for this streamer.
        n_samples_to_stream = None
        if self.rate is not None:
            n_samples_to_stream = 1 + self.rng.poisson(lam=self.rate)

        # instantiate a new streamer
        streamer = self.streamers[idx].iterate(max_iter=n_samples_to_stream)
        weight = self.weights[idx]

        # If we're sampling without replacement, zero this one out
        # This effectively disables this stream as soon as it is chosen,
        # preventing it from being chosen again (unless it is revived)
        # if not self.with_replacement:
        if self.mode != "with_replacement":
            self.distribution_[idx] = 0.0

            # Correct the distribution
            if (self.distribution_ > 0).any():
                self.distribution_[:] /= np.sum(self.distribution_)

        return streamer, weight