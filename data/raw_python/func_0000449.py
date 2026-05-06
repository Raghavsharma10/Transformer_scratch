def build(cls: Type[AN], node: ast.stmt) -> List[AN]:
        """
        Starting at this ``node``, check if it's an act node. If it's a context
        manager, recurse into child nodes.

        Returns:
            List of all act nodes found.
        """
        if node_is_result_assignment(node):
            return [cls(node, ActNodeType.result_assignment)]
        if node_is_pytest_raises(node):
            return [cls(node, ActNodeType.pytest_raises)]
        if node_is_unittest_raises(node):
            return [cls(node, ActNodeType.unittest_raises)]

        token = node.first_token  # type: ignore
        # Check if line marked with '# act'
        if token.line.strip().endswith('# act'):
            return [cls(node, ActNodeType.marked_act)]

        # Recurse (downwards) if it's a context manager
        if isinstance(node, ast.With):
            return cls.build_body(node.body)

        return []