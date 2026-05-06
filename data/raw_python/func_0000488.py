def build_act(cls: Type[_Block], node: ast.stmt, test_func_node: ast.FunctionDef) -> _Block:
        """
        Act block is a single node - either the act node itself, or the node
        that wraps the act node.
        """
        add_node_parents(test_func_node)
        # Walk up the parent nodes of the parent node to find test's definition.
        act_block_node = node
        while act_block_node.parent != test_func_node:  # type: ignore
            act_block_node = act_block_node.parent  # type: ignore
        return cls([act_block_node], LineType.act)