def add_widget(self, widget=None, row=None, col=None, row_span=1,
                   col_span=1, **kwargs):
        """
        Add a new widget to this grid. This will cause other widgets in the
        grid to be resized to make room for the new widget. Can be used
        to replace a widget as well

        Parameters
        ----------
        widget : Widget | None
            The Widget to add. New widget is constructed if widget is None.
        row : int
            The row in which to add the widget (0 is the topmost row)
        col : int
            The column in which to add the widget (0 is the leftmost column)
        row_span : int
            The number of rows to be occupied by this widget. Default is 1.
        col_span : int
            The number of columns to be occupied by this widget. Default is 1.
        **kwargs : dict
            parameters sent to the new Widget that is constructed if
            widget is None

        Notes
        -----
        The widget's parent is automatically set to this grid, and all other
        parent(s) are removed.
        """
        if row is None:
            row = self._next_cell[0]
        if col is None:
            col = self._next_cell[1]

        if widget is None:
            widget = Widget(**kwargs)
        else:
            if kwargs:
                raise ValueError("cannot send kwargs if widget is given")

        _row = self._cells.setdefault(row, {})
        _row[col] = widget
        self._grid_widgets[self._n_added] = (row, col, row_span, col_span,
                                             widget)
        self._n_added += 1
        widget.parent = self

        self._next_cell = [row, col+col_span]

        widget._var_w = Variable("w-(row: %s | col: %s)" % (row, col))
        widget._var_h = Variable("h-(row: %s | col: %s)" % (row, col))

        # update stretch based on colspan/rowspan
        # usually, if you make something consume more grids or columns,
        # you also want it to actually *take it up*, ratio wise.
        # otherwise, it will never *use* the extra rows and columns,
        # thereby collapsing the extras to 0.
        stretch = list(widget.stretch)
        stretch[0] = col_span if stretch[0] is None else stretch[0]
        stretch[1] = row_span if stretch[1] is None else stretch[1]
        widget.stretch = stretch

        self._need_solver_recreate = True

        return widget