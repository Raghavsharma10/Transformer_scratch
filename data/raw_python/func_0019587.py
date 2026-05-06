def _activate(self):
        """ShuffledMux's activate is similar to StochasticMux,
        but there is no 'n_active', since all the streams are always available.
        """
        self.streams_ = [None] * self.n_streams

        # Weights of the active streams.
        # Once a stream is exhausted, it is set to 0.
        # Upon activation, this is just a copy of self.weights.
        self.stream_weights_ = np.array(self.weights, dtype=float)
        # How many samples have been drawn from each (active) stream.
        self.stream_counts_ = np.zeros(self.n_streams, dtype=int)

        # Initialize each active stream.
        for idx in range(self.n_streams):
            # Setup a new streamer at this index.
            self._new_stream(idx)

        self.weight_norm_ = np.sum(self.stream_weights_)