def build_subtree_strut(self, result, *args, **kwargs):
        """
        Returns a dictionary in form of
        {node:Resource, children:{node_id: Resource}}

        :param result:
        :return:
        """
        items = list(result)
        root_elem = {"node": None, "children": OrderedDict()}
        if len(items) == 0:
            return root_elem
        for _, node in enumerate(items):
            new_elem = {"node": node.Resource, "children": OrderedDict()}
            path = list(map(int, node.path.split("/")))
            parent_node = root_elem
            normalized_path = path[:-1]
            if normalized_path:
                for path_part in normalized_path:
                    parent_node = parent_node["children"][path_part]
            parent_node["children"][new_elem["node"].resource_id] = new_elem
        return root_elem