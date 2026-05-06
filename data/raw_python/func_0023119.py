def detach(self, filt, view=None):
        """Detach a filter.

        Parameters
        ----------
        filt : object
            The filter to detach.
        view : instance of VisualView | None
            The view to use.
        """
        for v in self._subvisuals:
            v.detach(filt, v)