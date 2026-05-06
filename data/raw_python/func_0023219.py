def add_view(self, row=None, col=None, row_span=1, col_span=1,
                 **kwargs):
        """
        Create a new ViewBox and add it as a child widget.

        Parameters
        ----------
        row : int
            The row in which to add the widget (0 is the topmost row)
        col : int
            The column in which to add the widget (0 is the leftmost column)
        row_span : int
            The number of rows to be occupied by this widget. Default is 1.
        col_span : int
            The number of columns to be occupied by this widget. Default is 1.
        **kwargs : dict
            Keyword arguments to pass to `ViewBox`.
        """
        from .viewbox import ViewBox
        view = ViewBox(**kwargs)
        return self.add_widget(view, row, col, row_span, col_span)