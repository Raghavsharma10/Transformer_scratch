def set_mapping(self, x0, x1, update=True):
        """Configure this transform such that it maps points x0 => x1

        Parameters
        ----------
        x0 : array-like, shape (2, 2) or (2, 3)
            Start location.
        x1 : array-like, shape (2, 2) or (2, 3)
            End location.
        update : bool
            If False, then the update event is not emitted.

        Examples
        --------
        For example, if we wish to map the corners of a rectangle::

            >>> p1 = [[0, 0], [200, 300]]

        onto a unit cube::

            >>> p2 = [[-1, -1], [1, 1]]

        then we can generate the transform as follows::

            >>> tr = STTransform()
            >>> tr.set_mapping(p1, p2)
            >>> assert tr.map(p1)[:,:2] == p2  # test

        """
        # if args are Rect, convert to array first
        if isinstance(x0, Rect):
            x0 = x0._transform_in()[:3]
        if isinstance(x1, Rect):
            x1 = x1._transform_in()[:3]
        
        x0 = np.asarray(x0)
        x1 = np.asarray(x1)
        if (x0.ndim != 2 or x0.shape[0] != 2 or x1.ndim != 2 or 
                x1.shape[0] != 2):
            raise TypeError("set_mapping requires array inputs of shape "
                            "(2, N).")
        denom = x0[1] - x0[0]
        mask = denom == 0
        denom[mask] = 1.0
        s = (x1[1] - x1[0]) / denom
        s[mask] = 1.0
        s[x0[1] == x0[0]] = 1.0
        t = x1[0] - s * x0[0]
        s = as_vec4(s, default=(1, 1, 1, 1))
        t = as_vec4(t, default=(0, 0, 0, 0))
        self._set_st(scale=s, translate=t, update=update)