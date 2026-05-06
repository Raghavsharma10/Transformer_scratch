def remove_widget(self, widget):
        """
        Remove a Widget as a managed child of this Widget.

        Parameters
        ----------
        widget : instance of Widget
            The widget to remove.
        """
        self._widgets.remove(widget)
        widget.parent = None
        self._update_child_widgets()