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

        color : list, array or 4-tuple
           Path color

        linewidth : list, array or float
           Path linewidth

        antialias : list, array or float
           Path antialias area
        """

        itemsize = itemsize or len(P)
        itemcount = len(P) / itemsize

        P = P.reshape(itemcount, itemsize, 3)
        if closed:
            V = np.empty((itemcount, itemsize + 3), dtype=self.vtype)
            # Apply default values on vertices
            for name in self.vtype.names:
                if name not in ['collection_index', 'prev', 'curr', 'next']:
                    V[name][1:-2] = kwargs.get(name, self._defaults[name])
            V['prev'][:, 2:-1] = P
            V['prev'][:, 1] = V['prev'][:, -2]
            V['curr'][:, 1:-2] = P
            V['curr'][:, -2] = V['curr'][:, 1]
            V['next'][:, 0:-3] = P
            V['next'][:, -3] = V['next'][:, 0]
            V['next'][:, -2] = V['next'][:, 1]
        else:
            V = np.empty((itemcount, itemsize + 2), dtype=self.vtype)
            # Apply default values on vertices
            for name in self.vtype.names:
                if name not in ['collection_index', 'prev', 'curr', 'next']:
                    V[name][1:-1] = kwargs.get(name, self._defaults[name])
            V['prev'][:, 2:] = P
            V['prev'][:, 1] = V['prev'][:, 2]
            V['curr'][:, 1:-1] = P
            V['next'][:, :-2] = P
            V['next'][:, -2] = V['next'][:, -3]

        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]
        V = V.ravel()
        V = np.repeat(V, 2, axis=0)
        V['id'] = np.tile([1, -1], len(V) / 2)
        if closed:
            V = V.reshape(itemcount, 2 * (itemsize + 3))
        else:
            V = V.reshape(itemcount, 2 * (itemsize + 2))
        V["id"][:, :2] = 2, -2
        V["id"][:, -2:] = 2, -2
        V = V.ravel()

        # Uniforms
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ["__unused__"]:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None

        Collection.append(self, vertices=V, uniforms=U,
                          itemsize=2 * (itemsize + 2 + closed))