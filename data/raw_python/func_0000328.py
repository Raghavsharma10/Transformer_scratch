def depth_first(self, top_down=True):
        """
        Iterate depth-first.

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
            ...     print(node.name)
            ...
            a
            outer
            inner
            b
            c
            d

        ::

            >>> for node in root_container.depth_first(top_down=False):
            ...     print(node.name)
            ...
            a
            b
            c
            inner
            d
            outer

        """
        for child in tuple(self):
            if top_down:
                yield child
            if isinstance(child, UniqueTreeContainer):
                yield from child.depth_first(top_down=top_down)
            if not top_down:
                yield child