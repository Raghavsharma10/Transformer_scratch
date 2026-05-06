def odinweb_node_formatter(path_node):
        # type: (PathParam) -> str
        """
        Format a node to be consumable by the `UrlPath.parse`.
        """
        args = [path_node.name]
        if path_node.type:
            args.append(path_node.type.name)
        if path_node.type_args:
            args.append(path_node.type_args)
        return "{{{}}}".format(':'.join(args))