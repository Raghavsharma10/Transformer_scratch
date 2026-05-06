def _update_complex_list(self, **update_props):
        """ Default update operation for lists of complex structs """

        prop = update_props['prop']
        xpath_root = self._get_xroot_for(prop)
        xpath_map = self._data_structures[prop]

        return update_complex_list(xpath_root=xpath_root, xpath_map=xpath_map, **update_props)