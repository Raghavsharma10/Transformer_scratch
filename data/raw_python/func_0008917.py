def fix_edge_pixels(self, edge_init_data, edge_init_done, edge_init_todo):
        """
        This function fixes the pixels on the very edge of the tile.
        Drainage is calculated if the edge is downstream from the interior.
        If there is data available on the edge (from edge_init_data, for eg)
        then this data is used.

        This is a bit of hack to take care of the edge-values. It could
        possibly be handled through the main algorithm, but at least here
        the treatment is explicit.
        """
        data, dX, dY, direction, flats = \
            self.data, self.dX, self.dY, self.direction, self.flats
        sides = ['left', 'right', 'top', 'bottom']
        slices_o = [[slice(None), slice(1, 2)], [slice(None), slice(-2, -1)],
                    [slice(1, 2), slice(None)], [slice(-2, -1), slice(None)]]
        slices_d = [[slice(None), slice(0, 1)], [slice(None), slice(-1, None)],
                    [slice(0, 1), slice(None)], [slice(-1, None), slice(None)]]

        # The first set of edges will have contributions from two nodes whereas
        # the second set of edges will only have contributinos from one node
        indices = {'left': [[3, 4], [2, 5]], 'right': [[0, 7], [1, 6]],
                   'top': [[1, 2], [0, 3]], 'bottom': [[5, 6], [4, 7]]}

        # Figure out which section the drainage goes towards, and what
        # proportion goes to the straight-sided (as opposed to diagonal) node.

        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            section, proportion = \
                self._calc_uca_section_proportion(data[slice_o],
                                                  dX[slice_o[0]],
                                                  dY[slice_o[0]],
                                                  direction[slice_o],
                                                  flats[slice_o])
            # self-initialize:
            if side in ['left', 'right']:
                self.uca[slice_d] = \
                    np.concatenate(([dX[slice_d[0]][0] * dY[slice_d[0]][0]],
                                    dX[slice_d[0]] * dY[slice_d[0]]))\
                    .reshape(self.uca[slice_d].shape)
            else:
                self.uca[slice_d] = dX[slice_d[0]][0] * dY[slice_d[0]][0]
            for e in range(2):
                for i in indices[side][e]:
                    ed = self.facets[i][2]
                    ids = section == i
                    if e == 0:
                        self.uca[slice_d][ids] += self.uca[slice_o][ids] \
                            * proportion[ids]
                    self.uca[slice_d][ids] += \
                        np.roll(np.roll(self.uca[slice_o] * (1 - proportion),
                                        ed[0], 0),
                                ed[1], 1)[ids]
                    if e == 1:
                        self.uca[slice_d][ids] += \
                            np.roll(np.roll(self.uca[slice_o] * (proportion),
                                            ed[0], 0),
                                    ed[1], 1)[ids]

            # Finally, add the edge data from adjacent tiles
            if edge_init_done is not None:
                ids = edge_init_done[side]  # > 0
                if side in ['left', 'right']:
                    self.uca[slice_d][ids, :] = \
                        edge_init_data[side][ids][:, None]
                else:
                    self.uca[slice_d][:, ids] = edge_init_data[side][ids]