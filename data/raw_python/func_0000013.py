def debug_inspect_node(self, node_msindex):
        """
        Get info about the node. See pycut.inspect_node() for details.
        Processing is done in temporary shape.

        :param node_seed:
        :return: node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds
        """
        return inspect_node(self.nlinks, self.unariesalt2, self.msinds, node_msindex)