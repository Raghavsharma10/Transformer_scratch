def remove(self):
        """
        Remove the layer artist from the visualization
        """

        if self._multiscat is None:
            return

        self._multiscat.deallocate(self.id)
        self._multiscat = None

        self._viewer_state.remove_global_callback(self._update_scatter)
        self.state.remove_global_callback(self._update_scatter)