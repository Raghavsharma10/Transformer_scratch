def fixpoint(self, clamping, steps=0):
        """
        Computes the fixpoint with respect to a given :class:`caspo.core.clamping.Clamping`

        Parameters
        ----------
        clamping : :class:`caspo.core.clamping.Clamping`
            The clamping with respect to the fixpoint is computed

        steps : int
            If greater than zero, a maximum number of steps is performed. Otherwise
            it continues until reaching a fixpoint. Note that if no fixpoint exists,
            e.g. a network with a negative feedback-loop, this will never end unless
            you provide a maximum number of steps.

        Returns
        -------
        dict
            The key-value mapping describing the state of the logical network
        """
        current = dict.fromkeys(self.variables(), 0)
        updated = self.step(current, clamping)
        steps -= 1
        while current != updated and steps != 0:
            current = updated
            updated = self.step(current, clamping)

        return current