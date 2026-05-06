def _edge_group_substitution(
        self, ndid, nsplit, idxs, sr_tab, ndoffset, ed_remove, into_or_from
    ):
        """
        Reconnect edges.
        :param ndid: id of low resolution edges
        :param nsplit: number of split
        :param idxs: indexes of low resolution
        :param sr_tab:
        :param ndoffset:
        :param ed_remove:
        :param into_or_from: if zero, connection of input edges is done. If one, connection of output edges
        is performed.
        :return:
        """
        # this is useful for type(idxs) == np.ndarray
        eidxs = idxs[nm.where(self.edges[idxs, 1 - into_or_from] == ndid)[0]]
        # selected_edges = self.edges[idxs, 1 - into_or_from]
        # selected_edges == ndid
        # whre = nm.where(self.edges[idxs, 1 - into_or_from] == ndid)
        # whre0 = (nm.where(self.edges[idxs, 1 - into_or_from] == ndid) == ndid)[0]
        # eidxs = [idxs[i] for i in idxs]
        for igrp in self.edges_by_group(eidxs):
            if igrp.shape[0] > 1:
                # high resolution block to high resolution block
                # all directions are the same
                directions = self.edge_dir[igrp[0]]
                edge_indexes = sr_tab[directions, :].T.flatten() + ndoffset
                # debug code
                # if len(igrp) != len(edge_indexes):
                #     print("Problem ")
                self.edges[igrp, 1] = edge_indexes
                if self._edge_weight_table is not None:
                    self.edges_weights[igrp] = self._edge_weight_table[1, directions]
            else:
                # low res block to hi res block, if into_or_from is set to 0
                # hig res block to low res block, if into_or_from is set to 1
                ed_remove.append(igrp[0])
                # number of new edges is equal to number of pixels on one side of the box (in 2D and D too)
                nnewed = np.power(nsplit, self.data.ndim - 1)
                muleidxs = nm.tile(igrp, nnewed)
                # copy the low-res edge multipletime
                newed = self.edges[muleidxs, :]
                neweddir = self.edge_dir[muleidxs]
                local_node_ids = sr_tab[
                    self.edge_dir[igrp] + self.data.ndim * into_or_from, :
                ].T.flatten()
                # first or second (the actual) node id is substitued by new node indexes
                newed[:, 1 - into_or_from] = local_node_ids + ndoffset
                if self._edge_weight_table is not None:
                    self.add_edges(
                        newed, neweddir, self.edge_group[igrp], edge_low_or_high=1
                    )
                else:
                    self.add_edges(
                        newed, neweddir, self.edge_group[igrp], edge_low_or_high=None
                    )
        return ed_remove