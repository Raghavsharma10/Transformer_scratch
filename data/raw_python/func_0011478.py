def _get_node_name(self, node):
        """Get the name of the node - check for node.name and
        node.type.declname. Not sure why the second one occurs
        exactly - it happens with declaring a new struct field
        with parameters"""
        res = getattr(node, "name", None)
        if res is None:
            return res

        if isinstance(res, AST.TypeDecl):
            return res.declname
        
        return res