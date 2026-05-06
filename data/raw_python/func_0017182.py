def prune(self, nodes, preserve_branch_length=False):
        """
        Prunes the topology of a node to conserve only a selected list of leaf
        internal nodes. The minimum number of nodes that conserve the
        topological relationships among the requested nodes will be
        retained. Root node is always conserved.

        Parameters:
        -----------
        nodes:
            a list of node names or node objects that should be retained

        preserve_branch_length: 
            If True, branch lengths of the deleted nodes are transferred 
            (summed up) to its parent's branch, thus keeping original distances
            among nodes.

        **Examples:**
          t1 = Tree('(((((A,B)C)D,E)F,G)H,(I,J)K)root;', format=1)
          t1.prune(['A', 'B'])

          #                /-A
          #          /D /C|
          #       /F|      \-B
          #      |  |
          #    /H|   \-E
          #   |  |                        /-A
          #-root  \-G                 -root
          #   |                           \-B
          #   |   /-I
          #    \K|
          #       \-J

          t1 = Tree('(((((A,B)C)D,E)F,G)H,(I,J)K)root;', format=1)
          t1.prune(['A', 'B', 'C'])

          #                /-A
          #          /D /C|
          #       /F|      \-B
          #      |  |
          #    /H|   \-E
          #   |  |                              /-A
          #-root  \-G                  -root- C|
          #   |                                 \-B
          #   |   /-I
          #    \K|
          #       \-J

          t1 = Tree('(((((A,B)C)D,E)F,G)H,(I,J)K)root;', format=1)
          t1.prune(['A', 'B', 'I'])

          #                /-A
          #          /D /C|
          #       /F|      \-B
          #      |  |
          #    /H|   \-E                    /-I
          #   |  |                      -root
          #-root  \-G                      |   /-A
          #   |                             \C|
          #   |   /-I                          \-B
          #    \K|
          #       \-J

          t1 = Tree('(((((A,B)C)D,E)F,G)H,(I,J)K)root;', format=1)
          t1.prune(['A', 'B', 'F', 'H'])

          #                /-A
          #          /D /C|
          #       /F|      \-B
          #      |  |
          #    /H|   \-E
          #   |  |                              /-A
          #-root  \-G                -root-H /F|
          #   |                                 \-B
          #   |   /-I
          #    \K|
          #       \-J

        """

        def cmp_nodes(x, y):
            # if several nodes are in the same path of two kept nodes,
            # only one should be maintained. This prioritize internal
            # nodes that are already in the to_keep list and then
            # deeper nodes (closer to the leaves).
            if n2depth[x] > n2depth[y]:
                return -1
            elif n2depth[x] < n2depth[y]:
                return 1
            else:
                return 0

        to_keep = set(_translate_nodes(self, *nodes))
        start, node2path = self.get_common_ancestor(to_keep, get_path=True)
        to_keep.add(self)

        # Calculate which kept nodes are visiting the same nodes in
        # their path to the common ancestor.
        n2count = {}
        n2depth = {}
        for seed, path in six.iteritems(node2path):
            for visited_node in path:
                if visited_node not in n2depth:
                    depth = visited_node.get_distance(start, topology_only=True)
                    n2depth[visited_node] = depth
                if visited_node is not seed:
                    n2count.setdefault(visited_node, set()).add(seed)

        # if several internal nodes are in the path of exactly the same kept
        # nodes, only one (the deepest) should be maintain.
        visitors2nodes = {}
        for node, visitors in six.iteritems(n2count):
            # keep nodes connection at least two other nodes
            if len(visitors)>1:
                visitor_key = frozenset(visitors)
                visitors2nodes.setdefault(visitor_key, set()).add(node)

        for visitors, nodes in six.iteritems(visitors2nodes):
            if not (to_keep & nodes):
                sorted_nodes = sorted(nodes, key=cmp_to_key(cmp_nodes))
                to_keep.add(sorted_nodes[0])

        for n in self.get_descendants('postorder'):
            if n not in to_keep:
                if preserve_branch_length:
                    if len(n.children) == 1:
                        n.children[0].dist += n.dist
                    elif len(n.children) > 1 and n.up:
                        n.up.dist += n.dist

                n.delete(prevent_nondicotomic=False)