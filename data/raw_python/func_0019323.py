def old(self):
        """Assess to the state value(s) at beginning of the time step, which
        has been processed most recently.  When using *HydPy* in the
        normal manner.  But it can be helpful for demonstration and debugging
        purposes.
        """
        value = getattr(self.fastaccess_old, self.name, None)
        if value is None:
            raise RuntimeError(
                'No value/values of sequence %s has/have '
                'not been defined so far.'
                % objecttools.elementphrase(self))
        else:
            if self.NDIM:
                value = numpy.asarray(value)
            return value