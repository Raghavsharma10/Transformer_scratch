def parse(cls, url_path):
        # type: (str) -> UrlPath
        """
        Parse a string into a URL path (simple eg does not support typing of URL parameters)
        """
        if not url_path:
            return cls()

        nodes = []
        for node in url_path.rstrip('/').split('/'):
            # Identifies a PathNode
            if '{' in node or '}' in node:
                m = PATH_NODE_RE.match(node)
                if not m:
                    raise ValueError("Invalid path param: {}".format(node))

                # Parse out name and type
                name, param_type, param_arg = m.groups()
                try:
                    type_ = Type[param_type]
                except KeyError:
                    if param_type is not None:
                        raise ValueError("Unknown param type `{}` in: {}".format(param_type, node))
                    type_ = Type.Integer

                nodes.append(PathParam(name, type_, param_arg))
            else:
                nodes.append(node)

        return cls(*nodes)