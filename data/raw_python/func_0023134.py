def append(self, P0, P1, itemsize=None, **kwargs):
        """
        Append a new set of segments to the collection.

        For kwargs argument, n is the number of vertices (local) or the number
        of item (shared)

        Parameters
        ----------

        P : np.array
            Vertices positions of the path(s) to be added

        itemsize: int or None
            Size of an individual path

        caps : list, array or 2-tuple
           Path start /end cap

        color : list, array or 4-tuple
           Path color

        linewidth : list, array or float
           Path linewidth

        antialias : list, array or float
           Path antialias area
        """

        itemsize = itemsize or 1
        itemcount = len(P0) // itemsize

        V = np.empty(itemcount, dtype=self.vtype)

        # Apply default values on vertices
        for name in self.vtype.names:
            if name not in ['collection_index', 'P0', 'P1', 'index']:
                V[name] = kwargs.get(name, self._defaults[name])

        V['P0'] = P0
        V['P1'] = P1
        V = V.repeat(4, axis=0)
        V['index'] = np.resize([0, 1, 2, 3], 4 * itemcount * itemsize)

        I = np.ones((itemcount, 6), dtype=int)
        I[:] = 0, 1, 2, 0, 2, 3
        I[:] += 4 * np.arange(itemcount)[:, np.newaxis]
        I = I.ravel()

        # Uniforms
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ["__unused__"]:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None

        Collection.append(
            self, vertices=V, uniforms=U, indices=I, itemsize=4 * itemcount)