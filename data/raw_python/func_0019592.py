def _replace_stream(self, idx=None):
        """Called by `BaseMux`'s iterate() when a stream is exhausted.
        Set the stream to None so it is ignored once exhausted.

        Parameters
        ----------
        idx : int or None

        Raises
        ------
        StopIteration
            If all streams are consumed, and `mode`=="exahustive"
        """
        self.streams_[idx] = None

        # Check if we've now exhausted all the streams.
        if not self._streamers_available():
            if self.mode == 'exhaustive':
                pass

            elif self.mode == "cycle":
                self._setup_streams(permute=False)

            elif self.mode == "permuted_cycle":
                self._setup_streams(permute=True)