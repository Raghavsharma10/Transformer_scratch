def set_frustum(self, l, r, b, t, n, f):
        """Set the frustum

        Parameters
        ----------
        l : float
            Left.
        r : float
            Right.
        b : float
            Bottom.
        t : float
            Top.
        n : float
            Near.
        f : float
            Far.
        """
        self.matrix = transforms.frustum(l, r, b, t, n, f)