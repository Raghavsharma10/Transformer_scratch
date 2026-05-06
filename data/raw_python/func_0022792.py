def append(self, P, closed=False, itemsize=None, **kwargs):
        """
        Append a new set of vertices to the collection.

        For kwargs argument, n is the number of vertices (local) or the number
        of item (shared)

        Parameters
        ----------

        P : np.array
            Vertices positions of the path(s) to be added

        closed: bool
            Whether path(s) is/are closed

        itemsize: int or None
            Size of an individual path

        caps : list, array or 2-tuple
           Path start /end cap

        join : list, array or float
           path segment join

        color : list, array or 4-tuple
           Path color

        miter_limit : list, array or float
           Miter limit for join

        linewidth : list, array or float
           Path linewidth

        antialias : list, array or float
           Path antialias area
        """

        itemsize = itemsize or len(P)
        itemcount = len(P) / itemsize

        # Computes the adjacency information
        n, p = len(P), P.shape[-1]
        Z = np.tile(P, 2).reshape(2 * len(P), p)
        V = np.empty(n, dtype=self.vtype)

        V['p0'][1:-1] = Z[0::2][:-2]
        V['p1'][:-1] = Z[1::2][:-1]
        V['p2'][:-1] = Z[1::2][+1:]
        V['p3'][:-2] = Z[0::2][+2:]

        # Apply default values on vertices
        for name in self.vtype.names:
            if name not in ['collection_index', 'p0', 'p1', 'p2', 'p3']:
                V[name] = kwargs.get(name, self._defaults[name])

        # Extract relevant segments only
        V = (V.reshape(n / itemsize, itemsize)[:, :-1])
        if closed:
            V['p0'][:, 0] = V['p2'][:, -1]
            V['p3'][:, -1] = V['p1'][:, 0]
        else:
            V['p0'][:, 0] = V['p1'][:, 0]
            V['p3'][:, -1] = V['p2'][:, -1]
        V = V.ravel()

        # Quadruple each point (we're using 2 triangles / segment)
        # No shared vertices between segment because of joins
        V = np.repeat(V, 4, axis=0).reshape((len(V), 4))
        V['uv'] = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
        V = V.ravel()

        n = itemsize
        if closed:
            # uint16 for WebGL
            I = np.resize(
                np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), n * 2 * 3)
            I += np.repeat(4 * np.arange(n, dtype=np.uint32), 6)
            I[-6:] = 4 * n - 6, 4 * n - 5, 0, 4 * n - 5, 0, 1
        else:
            I = np.resize(
                np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), (n - 1) * 2 * 3)
            I += np.repeat(4 * np.arange(n - 1, dtype=np.uint32), 6)
        I = I.ravel()

        # Uniforms
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ["__unused__"]:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None

        Collection.append(self, vertices=V, uniforms=U,
                          indices=I, itemsize=itemsize * 4 - 4)