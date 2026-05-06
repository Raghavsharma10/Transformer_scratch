def _init_metadata(self):
        """
        Dynamically sets attributes from a Dictionary passed in by children.
        The Dictionary will contain the name of each attribute as keys, and
        either an XPATH mapping to a text value in _xml_tree, or a function
        that takes no parameters and returns the intended value.
        """

        if self._data_map is None:
            self._init_data_map()

        validate_properties(self._data_map, self._metadata_props)

        # Parse attribute values and assign them: key = parse(val)

        for prop in self._data_map:
            setattr(self, prop, parse_property(self._xml_tree, None, self._data_map, prop))

        self.has_data = any(getattr(self, prop) for prop in self._data_map)