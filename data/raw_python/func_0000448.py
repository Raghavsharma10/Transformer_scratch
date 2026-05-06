def build_body(cls: Type[AN], body: List[ast.stmt]) -> List:
        """
        Note:
            Return type is probably ``-> List[AN]``, but can't get it to pass.
        """
        act_nodes = []  # type: List[ActNode]
        for child_node in body:
            act_nodes += ActNode.build(child_node)
        return act_nodes