def _clrgen(n, h0, hr):
        """Default colour generating function

        Parameters
        ----------

        n : int
          Number of colours to generate
        h0 : float
          Initial H value in HSV colour specification
        hr : float
          Size of H value range to use for colour generation
          (final H value is h0 + hr)

        Returns
        -------
        clst : list of strings
          List of HSV format colour specification strings
        """

        n0 = n if n == 1 else n-1
        clst = ['%f,%f,%f' % (h0 + hr*hi/n0, 0.35, 0.85) for
                hi in range(n)]
        return clst