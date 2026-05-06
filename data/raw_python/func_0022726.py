def add_widget(self, widget):
        """
        Add a Widget as a managed child of this Widget.

        The child will be
        automatically positioned and sized to fill the entire space inside
        this Widget (unless _update_child_widgets is redefined).

        Parameters
        ----------
        widget : instance of Widget
            The widget to add.

        Returns
        -------
        widget : instance of Widget
            The widget.
        """
        self._widgets.append(widget)
        widget.parent = self
        self._update_child_widgets()
        return widget