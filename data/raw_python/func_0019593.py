def _new_stream(self):
        '''Grab the next stream from the input streamers, and start it.

        Raises
        ------
        StopIteration
            When the input list or generator of streamers is complete,
            will raise a StopIteration. If `mode == cycle`, it
            will instead restart iterating from the beginning of the sequence.
        '''
        try:
            # Advance the stream_generator_ to get the next available stream.
            # If successful, this will make self.chain_streamer_.active True
            next_stream = six.advance_iterator(self.stream_generator_)

        except StopIteration:
            # If running with cycle, restart the chain_streamer_ after
            # exhaustion.
            if self.mode == "cycle":
                self.stream_generator_ = self.chain_streamer_.iterate()

                # Try again to get the next stream;
                # if it fails this time, just let it raise the StopIteration;
                # this means the streams are probably dead or empty.
                next_stream = six.advance_iterator(self.stream_generator_)

            # If running in exhaustive mode
            else:
                # self.chain_streamer_ should no longer be active, so
                # the outer loop should fall out without running.
                next_stream = None

        if next_stream is not None:
            # Start that stream, and return it.
            streamer = next_stream.iterate()

            # Activate the Streamer
            self.streams_[0] = streamer

            # Reset the sample count to zero
            self.stream_counts_[0] = 0