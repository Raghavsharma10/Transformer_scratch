def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Don't activate the stream if the weight is 0 or None
        if self.stream_weights_[idx]:
            self.streams_[idx] = self.streamers[idx].iterate()
        else:
            self.streams_[idx] = None

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0