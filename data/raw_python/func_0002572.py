def _node_to_dict(cls, node, json, json_fields):
        """ Helper method for ``get_tree``.
        """
        if json:
            pk_name = node.get_pk_name()
            # jqTree or jsTree format
            result = {'id': getattr(node, pk_name), 'label': node.__repr__()}
            if json_fields:
                result.update(json_fields(node))
        else:
            result = {'node': node}
        return result