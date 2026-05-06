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
        if view is None:
            self._vshare.filters.append(filt)
            for view in self._vshare.views.keys():
                filt._attach(view)
        else:
            view._filters.append(filt)
            filt._attach(view)