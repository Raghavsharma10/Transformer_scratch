def append(self, P0, P1, itemsize=None, **kwargs):
        """
        Append a new set of segments to the collection.

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

        color : list, array or 4-tuple
           Path color
        """

        itemsize = itemsize or 1
        itemcount = len(P0) / itemsize

        V = np.empty(itemcount, dtype=self.vtype)

        # Apply default values on vertices
        for name in self.vtype.names:
            if name not in ['collection_index', 'P']:
                V[name] = kwargs.get(name, self._defaults[name])

        V = np.repeat(V, 2, axis=0)
        V['P'][0::2] = P0
        V['P'][1::2] = P1

        # Uniforms
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ["__unused__"]:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None

        Collection.append(self, vertices=V, uniforms=U, itemsize=itemsize)