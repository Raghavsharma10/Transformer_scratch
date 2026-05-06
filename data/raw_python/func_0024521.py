def add_atom(self, atom, _map=None):
        """
        new atom addition
        """
        if _map is None:
            _map = max(self, default=0) + 1
        elif _map in self._node:
            raise KeyError('atom with same number exists')

        attr_dict = self.node_attr_dict_factory()
        if isinstance(atom, str):
            attr_dict.element = atom
        elif isinstance(atom, int):
            attr_dict.element = elements_list[atom - 1]
        else:
            attr_dict.update(atom)

        self._adj[_map] = self.adjlist_inner_dict_factory()
        self._node[_map] = attr_dict
        self.flush_cache()
        return _map