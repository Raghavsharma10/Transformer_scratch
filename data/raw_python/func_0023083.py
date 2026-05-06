def set_ortho(self, l, r, b, t, n, f):
        """Set ortho transform

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
        self.matrix = transforms.ortho(l, r, b, t, n, f)