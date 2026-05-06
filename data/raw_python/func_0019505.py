def iterate(self, max_iter=None):
        '''Instantiate an iterator.

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If ``None``, exhaust the stream.

        Yields
        ------
        obj : Objects yielded by the streamer provided on init.

        See Also
        --------
        cycle : force an infinite stream.

        '''
        # Use self as context manager / calls __enter__() => _activate()
        with self as active_streamer:
            for n, obj in enumerate(active_streamer.stream_):
                if max_iter is not None and n >= max_iter:
                    break
                yield obj