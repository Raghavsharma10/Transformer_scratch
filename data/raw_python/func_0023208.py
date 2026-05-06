def append(self, vertices, uniforms=None, indices=None, itemsize=None):
        """
        Parameters
        ----------

        vertices : numpy array
            An array whose dtype is compatible with self.vdtype

        uniforms: numpy array
            An array whose dtype is compatible with self.utype

        indices : numpy array
            An array whose dtype is compatible with self.idtype
            All index values must be between 0 and len(vertices)

        itemsize: int, tuple or 1-D array
            If `itemsize is an integer, N, the array will be divided
            into elements of size N. If such partition is not possible,
            an error is raised.

            If `itemsize` is 1-D array, the array will be divided into
            elements whose succesive sizes will be picked from itemsize.
            If the sum of itemsize values is different from array size,
            an error is raised.
        """

        # Vertices
        # -----------------------------
        vertices = np.array(vertices).astype(self.vtype).ravel()
        vsize = self._vertices_list.size

        # No itemsize given
        # -----------------
        if itemsize is None:
            index = 0
            count = 1

        # Uniform itemsize (int)
        # ----------------------
        elif isinstance(itemsize, int):
            count = len(vertices) / itemsize
            index = np.repeat(np.arange(count), itemsize)

        # Individual itemsize (array)
        # ---------------------------
        elif isinstance(itemsize, (np.ndarray, list)):
            count = len(itemsize)
            index = np.repeat(np.arange(count), itemsize)
        else:
            raise ValueError("Itemsize not understood")

        if self.utype:
            vertices["collection_index"] = index + len(self)
        self._vertices_list.append(vertices, itemsize)

        # Indices
        # -----------------------------
        if self.itype is not None:
            # No indices given (-> automatic generation)
            if indices is None:
                indices = vsize + np.arange(len(vertices))
                self._indices_list.append(indices, itemsize)

            # Indices given
            # FIXME: variables indices (list of list or ArrayList)
            else:
                if itemsize is None:
                    I = np.array(indices) + vsize
                elif isinstance(itemsize, int):
                    I = vsize + (np.tile(indices, count) +
                                 itemsize * np.repeat(np.arange(count), len(indices)))  # noqa
                else:
                    raise ValueError("Indices not compatible with items")
                self._indices_list.append(I, len(indices))

        # Uniforms
        # -----------------------------
        if self.utype:
            if uniforms is None:
                uniforms = np.zeros(count, dtype=self.utype)
            else:
                uniforms = np.array(uniforms).astype(self.utype).ravel()
            self._uniforms_list.append(uniforms, itemsize=1)

        self._need_update = True