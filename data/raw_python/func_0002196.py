def projection_box(self, min_x, min_y, max_x, max_y):
        """Add a bounding box in projected (native) coordinates to the query.

        This adds a request for a spatial bounding box, bounded by (`min_x`, `max_x`) for
        x direction and (`min_y`, `max_y`) for the y direction. This modifies the query
        in-place, but returns ``self`` so that multiple queries can be chained together
        on one line.

        This replaces any existing spatial queries that have been set.

        Parameters
        ----------
        min_x : float
            The left edge of the bounding box
        min_y : float
            The bottom edge of the bounding box
        max_x : float
            The right edge of the bounding box
        max_y: float
            The top edge of the bounding box

        Returns
        -------
        self : NCSSQuery
            Returns self for chaining calls

        """
        self._set_query(self.spatial_query, minx=min_x, miny=min_y,
                        maxx=max_x, maxy=max_y)
        return self