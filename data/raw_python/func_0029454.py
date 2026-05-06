def validate(self):
        """ Default validation for updated properties: MAY be overridden in children """

        validate_properties(self._data_map, self._metadata_props)

        for prop in self._data_map:
            validate_any(prop, getattr(self, prop), self._data_structures.get(prop))

        return self