def append(self, points, indices, **kwargs):
        """
        Append a new set of vertices to the collection.

        For kwargs argument, n is the number of vertices (local) or the number
        of item (shared)

        Parameters
        ----------

        points : np.array
            Vertices composing the triangles

        indices : np.array
            Indices describing triangles

        color : list, array or 4-tuple
           Path color
        """

        itemsize = len(points)
        itemcount = 1

        V = np.empty(itemcount * itemsize, dtype=self.vtype)
        for name in self.vtype.names:
            if name not in ['collection_index', 'position']:
                V[name] = kwargs.get(name, self._defaults[name])
        V["position"] = points

        # Uniforms
        if self.utype:
            U = np.zeros(itemcount, dtype=self.utype)
            for name in self.utype.names:
                if name not in ["__unused__"]:
                    U[name] = kwargs.get(name, self._defaults[name])
        else:
            U = None

        I = np.array(indices).ravel()

        Collection.append(self, vertices=V, uniforms=U, indices=I,
                          itemsize=itemsize)