def _delete_subtree(self, nodes):
        """
        Delete subtree private method.

        No argument validation and usage of getter/setter private methods is
        used for speed
        """
        nodes = nodes if isinstance(nodes, list) else [nodes]
        iobj = [
            (self._db[node]["parent"], node)
            for node in nodes
            if self._node_name_in_tree(node)
        ]
        for parent, node in iobj:
            # Delete link to parent (if not root node)
            del_list = self._get_subtree(node)
            if parent:
                self._db[parent]["children"].remove(node)
            # Delete children (sub-tree)
            for child in del_list:
                del self._db[child]
            if self._empty_tree():
                self._root = None
                self._root_hierarchy_length = None