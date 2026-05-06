def build_assert(cls: Type[_Block], nodes: List[ast.stmt], min_line_number: int) -> _Block:
        """
        Assert block is all nodes that are after the Act node.

        Note:
            The filtering is *still* running off the line number of the Act
            node, when instead it should be using the last line of the Act
            block.
        """
        return cls(filter_assert_nodes(nodes, min_line_number), LineType._assert)