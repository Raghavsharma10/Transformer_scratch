def graph_order(self):
        """
        Get graph-order tuple for node.

        ::

            >>> from uqbar.containers import UniqueTreeContainer, UniqueTreeNode
            >>> root_container = UniqueTreeContainer(name="root")
            >>> outer_container = UniqueTreeContainer(name="outer")
            >>> inner_container = UniqueTreeContainer(name="inner")
            >>> node_a = UniqueTreeNode(name="a")
            >>> node_b = UniqueTreeNode(name="b")
            >>> node_c = UniqueTreeNode(name="c")
            >>> node_d = UniqueTreeNode(name="d")
            >>> root_container.extend([node_a, outer_container])
            >>> outer_container.extend([inner_container, node_d])
            >>> inner_container.extend([node_b, node_c])

        ::

            >>> for node in root_container.depth_first():
            ...     print(node.name, node.graph_order)
            ...
            a (0,)
            outer (1,)
            inner (1, 0)
            b (1, 0, 0)
            c (1, 0, 1)
            d (1, 1)

        """
        parentage = tuple(reversed(self.parentage))
        graph_order = []
        for i in range(len(parentage) - 1):
            parent, child = parentage[i : i + 2]
            graph_order.append(parent.index(child))
        return tuple(graph_order)