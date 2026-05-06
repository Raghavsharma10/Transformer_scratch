def _search_tree(self, name):
        """Search_tree for nodes that contain a specific hierarchy name."""
        tpl1 = "{sep}{name}{sep}".format(sep=self._node_separator, name=name)
        tpl2 = "{sep}{name}".format(sep=self._node_separator, name=name)
        tpl3 = "{name}{sep}".format(sep=self._node_separator, name=name)
        return sorted(
            [
                node
                for node in self._db
                if (tpl1 in node)
                or node.endswith(tpl2)
                or node.startswith(tpl3)
                or (name == node)
            ]
        )