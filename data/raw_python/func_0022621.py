def getvalues(self):
        """Yields all the values from 'generator_func' and type-checks.

        Yields:
            Whatever 'generator_func' yields.

        Raises:
            TypeError: if subsequent values are of a different type than first
                value.

            ValueError: if subsequent iteration returns a different number of
                values than the first iteration over the generator. (This would
                mean 'generator_func' is not stable.)
        """
        idx = 0
        generator = self._generator_func()
        first_value = next(generator)
        self._value_type = type(first_value)
        yield first_value

        for idx, value in enumerate(generator):
            if not isinstance(value, self._value_type):
                raise TypeError(
                    "All values of a repeated var must be of the same type."
                    " First argument was of type %r, but argument %r is of"
                    " type %r." %
                    (self._value_type, value, repeated.value_type(value)))

            self._watermark = max(self._watermark, idx + 1)
            yield value

        # Iteration stopped - check if we're at the previous watermark and raise
        # if not.
        if idx + 1 < self._watermark:
            raise ValueError(
                "LazyRepetition %r was previously able to iterate its"
                " generator up to idx %d, but this time iteration stopped after"
                " idx %d! Generator function %r is not stable." %
                (self, self._watermark, idx + 1, self._generator_func))

        # Watermark is higher than previous count! Generator function returned
        # more values this time than last time.
        if self._count is not None and self._watermark >= self._count:
            raise ValueError(
                "LazyRepetition %r previously iterated only up to idx %d but"
                " was now able to reach idx %d! Generator function %r is not"
                " stable." %
                (self, self._count - 1, idx + 1, self._generator_func))

        # We've finished iteration - cache count. After this the count will be
        # watermark + 1 forever.
        self._count = self._watermark + 1