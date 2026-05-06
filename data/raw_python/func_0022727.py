def add_grid(self, *args, **kwargs):
        """
        Create a new Grid and add it as a child widget.

        All arguments are given to Grid().
        """
        from .grid import Grid
        grid = Grid(*args, **kwargs)
        return self.add_widget(grid)