def bake(self, P, key='curr', closed=False, itemsize=None):
        """
        Given a path P, return the baked vertices as they should be copied in
        the collection if the path has already been appended.

        Example:
        --------

        paths.append(P)
        P *= 2
        paths['prev'][0] = bake(P,'prev')
        paths['curr'][0] = bake(P,'curr')
        paths['next'][0] = bake(P,'next')
        """

        itemsize = itemsize or len(P)
        itemcount = len(P) / itemsize  # noqa
        n = itemsize

        if closed:
            I = np.arange(n + 3)
            if key == 'prev':
                I -= 2
                I[0], I[1], I[-1] = n - 1, n - 1, n - 1
            elif key == 'next':
                I[0], I[-3], I[-2], I[-1] = 1, 0, 1, 1
            else:
                I -= 1
                I[0], I[-1], I[n + 1] = 0, 0, 0
        else:
            I = np.arange(n + 2)
            if key == 'prev':
                I -= 2
                I[0], I[1], I[-1] = 0, 0, n - 2
            elif key == 'next':
                I[0], I[-1], I[-2] = 1, n - 1, n - 1
            else:
                I -= 1
                I[0], I[-1] = 0, n - 1
        I = np.repeat(I, 2)
        return P[I]