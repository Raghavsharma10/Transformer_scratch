def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self.rng.choice(
            self.n_streams, p=self.distribution_)

        # Activate the Streamer, and get the weights
        self.streams_[idx], self.stream_weights_[idx] = self._activate_stream(
            self.stream_idxs_[idx])

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0