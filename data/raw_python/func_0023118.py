def attach(self, filt, view=None):
        """Attach a Filter to this visual

        Each filter modifies the appearance or behavior of the visual.

        Parameters
        ----------
        filt : object
            The filter to attach.
        view : instance of VisualView | None
            The view to use.
        """
        for v in self._subvisuals:
            v.attach(filt, v)