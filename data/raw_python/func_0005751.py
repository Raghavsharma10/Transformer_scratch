def _set_random_state_from(self, other):
        """
        Transfer the internal state from `other` to `self`.
        After this call, `self` will produce the same elements
        in the same order as `other` (even though they otherwise
        remain completely independent).
        """
        try:
            # this works if randgen is an instance of random.Random()
            self.randgen.setstate(other.randgen.getstate())
        except AttributeError:
            # this works if randgen is an instance of numpy.random.RandomState()
            self.randgen.set_state(other.randgen.get_state())

        return self