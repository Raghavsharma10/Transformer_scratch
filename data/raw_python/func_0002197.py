def strides(self, time=None, spatial=None):
        """Set time and/or spatial (horizontal) strides.

        This is only used on grid requests. Used to skip points in the returned data.
        This modifies the query in-place, but returns `self` so that multiple queries
        can be chained together on one line.

        Parameters
        ----------
        time : int, optional
            Stride for times returned. Defaults to None, which is equivalent to 1.
        spatial : int, optional
            Stride for horizontal grid. Defaults to None, which is equivalent to 1.

        Returns
        -------
        self : NCSSQuery
            Returns self for chaining calls

        """
        if time:
            self.add_query_parameter(timeStride=time)
        if spatial:
            self.add_query_parameter(horizStride=spatial)
        return self