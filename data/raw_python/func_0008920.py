def _mk_adjacency_matrix(self, section, proportion, flats, elev, mag, dX, dY):
        """
        Calculates the adjacency of connectivity matrix. This matrix tells
        which pixels drain to which.

        For example, the pixel i, will recieve area from np.nonzero(A[i, :])
        at the proportions given in A[i, :]. So, the row gives the pixel
        drain to, and the columns the pixels drained from.
        """
        shp = section.shape
        mat_data = np.row_stack((proportion, 1 - proportion))
        NN = np.prod(shp)
        i12 = np.arange(NN).reshape(shp)
        j1 = - np.ones_like(i12)
        j2 = - np.ones_like(i12)

        # make the connectivity for the non-flats/pits
        j1, j2 = self._mk_connectivity(section, i12, j1, j2)
        j = np.row_stack((j1, j2))
        i = np.row_stack((i12, i12))
        
        # connectivity for flats/pits
        if self.drain_pits:
            pit_i, pit_j, pit_prop, flats, mag = \
                self._mk_connectivity_pits(i12, flats, elev, mag, dX, dY)

            j = np.concatenate([j.ravel(), pit_j]).astype('int64')
            i = np.concatenate([i.ravel(), pit_i]).astype('int64')
            mat_data = np.concatenate([mat_data.ravel(), pit_prop])

        elif self.drain_flats:
            j1, j2, mat_data, flat_i, flat_j, flat_prop = \
                self._mk_connectivity_flats(
                    i12, j1, j2, mat_data, flats, elev, mag)

            j = np.concatenate([j.ravel(), flat_j]).astype('int64')
            i = np.concatenate([i.ravel(), flat_j]).astype('int64')
            mat_data = np.concatenate([mat_data.ravel(), flat_prop])



        # This prevents no-data values, remove connections when not present,
        # and makes sure that floating point precision errors do not
        # create circular references where a lower elevation cell drains
        # to a higher elevation cell
        I = ~np.isnan(mat_data) & (j != -1) & (mat_data > 1e-8) \
            & (elev.ravel()[j] <= elev.ravel()[i])

        mat_data = mat_data[I]
        j = j[I]
        i = i[I]

        # %%Make the matrix and initialize
        # What is A? The row i area receives area contributions from the
        # entries in its columns. If all the entries in my columns have
        #  drained, then I can drain.
        A = sps.csc_matrix((mat_data.ravel(),
                            np.row_stack((j.ravel(), i.ravel()))),
                           shape=(NN, NN))
        normalize = np.array(A.sum(0) + 1e-16).squeeze()
        A = np.dot(A, sps.diags(1/normalize, 0))

        return A