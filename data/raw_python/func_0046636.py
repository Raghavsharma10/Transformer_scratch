def materialize_node(self, node, uri, content, meta=None):
        """
        Set node uri and content from backend
        """
        node.uri = uri
        node.content = content
        node.meta = meta if meta is not None else {}