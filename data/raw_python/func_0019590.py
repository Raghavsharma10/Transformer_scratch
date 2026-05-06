def _next_sample_index(self):
        """Rotates through each active sampler by incrementing the index"""
        # Return the next streamer index where the streamer is not None,
        # wrapping around.
        idx = self.active_index_
        self.active_index_ += 1

        if self.active_index_ >= len(self.streams_):
            self.active_index_ = 0

        # Continue to increment if this streamer is exhausted (None)
        # This should never be infinite looping;
        # the `_streamers_available` check happens immediately
        # before this, so there should always be at least one not-None
        # streamer.
        while self.streams_[idx] is None:
            idx = self.active_index_
            self.active_index_ += 1

            if self.active_index_ >= len(self.streams_):
                self.active_index_ = 0

        return idx