def remove_widget(self, widget):
        """Remove a widget from this grid

        Parameters
        ----------
        widget : Widget
            The Widget to remove
        """

        self._grid_widgets = dict((key, val)
                                  for (key, val) in self._grid_widgets.items()
                                  if val[-1] != widget)

        self._need_solver_recreate = True