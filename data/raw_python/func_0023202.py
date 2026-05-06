def link_view(self, view):
        """Link this axis to a ViewBox

        This makes it so that the axis's domain always matches the
        visible range in the ViewBox.

        Parameters
        ----------
        view : instance of ViewBox
            The ViewBox to link.
        """
        if view is self._linked_view:
            return
        if self._linked_view is not None:
            self._linked_view.scene.transform.changed.disconnect(
                self._view_changed)
        self._linked_view = view
        view.scene.transform.changed.connect(self._view_changed)
        self._view_changed()