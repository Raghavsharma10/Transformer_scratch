def get_farthest_leaf(self, topology_only=False, is_leaf_fn=None):
        """
        Returns node's farthest descendant node (which is always a leaf), and the
        distance to it.

        :argument False topology_only: If set to True, distance
          between nodes will be referred to the number of nodes
          between them. In other words, topological distance will be
          used instead of branch length distances.

        :return: A tuple containing the farthest leaf referred to the
          current node and the distance to it.
        """
        min_node, min_dist, max_node, max_dist = self._get_farthest_and_closest_leaves(
        topology_only=topology_only, is_leaf_fn=is_leaf_fn)
        return max_node, max_dist