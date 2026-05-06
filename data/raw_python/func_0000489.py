def build_arrange(cls: Type[_Block], nodes: List[ast.stmt], max_line_number: int) -> _Block:
        """
        Arrange block is all non-pass and non-docstring nodes before the Act
        block start.
        """
        return cls(filter_arrange_nodes(nodes, max_line_number), LineType.arrange)