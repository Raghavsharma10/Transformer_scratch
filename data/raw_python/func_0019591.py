def _new_stream(self, idx):
        """Activate a new stream, given the index into the stream pool.

        BaseMux's _new_stream simply chooses a new stream and activates it.
        For special behavior (ie Weighted streams), you must override this
        in a child class.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        """
        # Get the stream index from the candidate pool
        stream_index = self.stream_idxs_[idx]

        # Activate the Streamer, and get the weights
        self.streams_[idx] = self.streamers[stream_index].iterate()

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0