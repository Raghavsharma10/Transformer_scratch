def memory_usage(self, string=False):
        """
        Get the memory usage estimate of the container.

        Args:
            string (bool): Human readable string (default false)

        See Also:
            :func:`~exa.core.container.Container.info`
        """
        if string:
            n = getsizeof(self)
            return ' '.join((str(s) for s in convert_bytes(n)))
        return self.info()['size']