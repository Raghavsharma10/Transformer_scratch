def _update_complex(self, **update_props):
        """ Default update operation for a complex struct """

        prop = update_props['prop']
        xpath_root = self._get_xroot_for(prop)
        xpath_map = self._data_structures[prop]

        return update_complex(xpath_root=xpath_root, xpath_map=xpath_map, **update_props)