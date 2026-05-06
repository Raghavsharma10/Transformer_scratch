def iterate(self, max_iter=None):
        """Yields items from the mux, and handles stream exhaustion and
        replacement.
        """
        if max_iter is None:
            max_iter = np.inf

        # Calls Streamer's __enter__, which calls activate()
        with self as active_mux:
            # Main sampling loop
            n = 0

            while n < max_iter and active_mux._streamers_available():
                # Pick a stream from the active set
                idx = active_mux._next_sample_index()

                # Can we sample from it?
                try:
                    # Then yield the sample
                    yield six.advance_iterator(active_mux.streams_[idx])

                    # Increment the sample counter
                    n += 1
                    active_mux.stream_counts_[idx] += 1

                except StopIteration:
                    # Oops, this stream is exhausted.

                    # Call child-class exhausted-stream behavior
                    active_mux._on_stream_exhausted(idx)

                    # Setup a new stream for this index
                    active_mux._replace_stream(idx)