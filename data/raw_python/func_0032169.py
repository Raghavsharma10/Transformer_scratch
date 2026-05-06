def draw(self, **kwargs):
        """Draw the polygon

        Optional Inputs:
        ------------
        All optional inputs are passed to ``matplotlib.patches.Polygon``

        Notes:
        ---------
        Does not accept maptype as an argument.
        """

        ax = mp.gca()
        shape = matplotlib.patches.Polygon(self.polygon, **kwargs)
        ax.add_artist(shape)