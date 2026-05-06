def _parse_complex_list(self, prop):
        """ Default parsing operation for lists of complex structs """

        xpath_root = self._get_xroot_for(prop)
        xpath_map = self._data_structures[prop]

        return parse_complex_list(self._xml_tree, xpath_root, xpath_map, prop)