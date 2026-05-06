def nodes(self):
        """A |Nodes| collection of all required nodes.

        >>> from hydpy import RiverBasinNumbers2Selection
        >>> rbns2s = RiverBasinNumbers2Selection(
        ...                            (111, 113, 1129, 11269, 1125, 11261,
        ...                             11262, 1123, 1124, 1122, 1121))

        Note that the required outlet node is added:

        >>> rbns2s.nodes
        Nodes("node_1123", "node_1125", "node_11269", "node_1129", "node_113",
              "node_outlet")

        It is both possible to change the prefix names of the nodes and
        the name of the outlet node separately:

        >>> rbns2s.node_prefix = 'b_'
        >>> rbns2s.last_node = 'l_node'
        >>> rbns2s.nodes
        Nodes("b_1123", "b_1125", "b_11269", "b_1129", "b_113", "l_node")
        """
        return (
            devicetools.Nodes(
                self.node_prefix+routers for routers in self._router_numbers) +
            devicetools.Node(self.last_node))