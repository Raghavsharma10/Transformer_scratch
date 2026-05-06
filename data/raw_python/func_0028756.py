def output_keys(self, source_keys):
        """
        Given input chunk keys, compute what keys will be needed to put
        the result into the result array.

        As an example of where this gets used - when we aggregate on a
        particular axis, the source keys may be ``(0:2, None:None)``, but for
        an aggregation on axis 0, they would result in target values on
        dimension 2 only and so be ``(None: None, )``.

        """
        keys = list(source_keys)
        # Remove the aggregated axis from the keys.
        del keys[self.axis]
        return tuple(keys)