def _parse_complex(self, prop):
        """ Default parsing operation for a complex struct """

        xpath_root = None
        xpath_map = self._data_structures[prop]

        return parse_complex(self._xml_tree, xpath_root, xpath_map, prop)