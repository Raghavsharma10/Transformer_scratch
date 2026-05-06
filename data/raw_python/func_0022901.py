def _compute_iso_line(self):
        """ compute LineVisual vertices, connects and color-index
        """
        level_index = []
        connects = []
        verts = []

        # calculate which level are within data range
        # this works for now and the existing examples, but should be tested
        # thoroughly also with the data-sanity check in set_data-function
        choice = np.nonzero((self.levels > self._data.min()) &
                            (self._levels < self._data.max()))
        levels_to_calc = np.array(self.levels)[choice]

        # save minimum level index
        self._level_min = choice[0][0]

        for level in levels_to_calc:
            # if we use matplotlib isoline algorithm we need to add half a
            # pixel in both (x,y) dimensions because isolines are aligned to
            # pixel centers
            if _HAS_MPL:
                nlist = self._iso.trace(level, level, 0)
                paths = nlist[:len(nlist)//2]
                v, c = self._get_verts_and_connect(paths)
                v += np.array([0.5, 0.5])
            else:
                paths = isocurve(self._data.astype(float).T, level,
                                 extend_to_edge=True, connected=True)
                v, c = self._get_verts_and_connect(paths)

            level_index.append(v.shape[0])
            connects.append(np.hstack((c, [False])))
            verts.append(v)

        self._li = np.hstack(level_index)
        self._connect = np.hstack(connects)
        self._verts = np.vstack(verts)