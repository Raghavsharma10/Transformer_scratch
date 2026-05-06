def cycle(self, max_iter=None):
        '''Iterate from the streamer infinitely.

        This function will force an infinite stream, restarting
        the streamer even if a StopIteration is raised.

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If `None`, iterate indefinitely.

        Yields
        ------
        obj : Objects yielded by the streamer provided on init.
        '''

        count = 0
        while True:
            for obj in self.iterate():
                count += 1
                if max_iter is not None and count > max_iter:
                    return
                yield obj