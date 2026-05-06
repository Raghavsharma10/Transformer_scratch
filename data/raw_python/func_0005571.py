def _rename_node(self, name, new_name):
        """
        Rename node private method.

        No argument validation and usage of getter/setter private methods is
        used for speed
        """
        # Update parent
        if not self.is_root(name):
            parent = self._db[name]["parent"]
            self._db[parent]["children"].remove(name)
            self._db[parent]["children"] = sorted(
                self._db[parent]["children"] + [new_name]
            )
        # Update children
        iobj = self._get_subtree(name) if name != self.root_name else self.nodes
        for key in iobj:
            new_key = key.replace(name, new_name, 1)
            new_parent = (
                self._db[key]["parent"]
                if key == name
                else self._db[key]["parent"].replace(name, new_name, 1)
            )
            self._db[new_key] = {
                "parent": new_parent,
                "children": [
                    child.replace(name, new_name, 1)
                    for child in self._db[key]["children"]
                ],
                "data": copy.deepcopy(self._db[key]["data"]),
            }
            del self._db[key]
        if name == self.root_name:
            self._root = new_name
            self._root_hierarchy_length = len(
                self.root_name.split(self._node_separator)
            )