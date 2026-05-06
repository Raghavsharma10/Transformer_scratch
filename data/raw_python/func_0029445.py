def _get_template(self, root=None, **metadata_defaults):
        """ Iterate over items metadata_defaults {prop: val, ...} to populate template """

        if root is None:
            if self._data_map is None:
                self._init_data_map()

            root = self._xml_root = self._data_map['_root']

        template_tree = self._xml_tree = create_element_tree(root)

        for prop, val in iteritems(metadata_defaults):
            path = self._data_map.get(prop)
            if path and val:
                setattr(self, prop, val)
                update_property(template_tree, None, path, prop, val)

        return template_tree