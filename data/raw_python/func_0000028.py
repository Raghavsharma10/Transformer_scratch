def add_edges(self, conn, edge_direction, edge_group=None, edge_low_or_high=None):
        """
        Add new edges at the end of the list.
        :param edge_direction: direction flag
        :param edge_group: describes group of edges from same low super node and same direction
        :param edge_low_or_high: zero for low to low resolution, one for high to high or high to low resolution.
        It is used to set weight from weight table.
        """
        last = self.lastedge
        if type(conn) is nm.ndarray:
            nadd = conn.shape[0]
            idx = slice(last, last + nadd)
            if edge_group is None:
                edge_group = nm.arange(nadd) + last
        else:
            nadd = 1
            idx = nm.array([last])
            conn = nm.array(conn).reshape((1, 2))
            if edge_group is None:
                edge_group = idx

        self.edges[idx, :] = conn
        self.edge_flag[idx] = True
        # t_start0 = time.time()
        # self.edge_flag_idx.extend(list(range(idx.start, idx.stop)))
        # self.stats["t split 082"] += time.time() - t_start0
        self.edge_dir[idx] = edge_direction
        self.edge_group[idx] = edge_group
        # TODO change this just to array of low_or_high_resolution
        if edge_low_or_high is not None and self._edge_weight_table is not None:
            self.edges_weights[idx] = self._edge_weight_table[
                edge_low_or_high, edge_direction
            ]
        self.lastedge += nadd
        self.nedges += nadd