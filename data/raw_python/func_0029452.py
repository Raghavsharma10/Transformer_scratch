def _update_dates(self, xpath_root=None, **update_props):
        """
        Default update operation for Dates metadata
        :see: gis_metadata.utils._complex_definitions[DATES]
        """

        tree_to_update = update_props['tree_to_update']
        prop = update_props['prop']
        values = (update_props['values'] or {}).get(DATE_VALUES) or u''
        xpaths = self._data_structures[prop]

        if not self.dates:
            date_xpaths = xpath_root
        elif self.dates[DATE_TYPE] != DATE_TYPE_RANGE:
            date_xpaths = xpaths.get(self.dates[DATE_TYPE], u'')
        else:
            date_xpaths = [
                xpaths[DATE_TYPE_RANGE_BEGIN],
                xpaths[DATE_TYPE_RANGE_END]
            ]

        if xpath_root:
            remove_element(tree_to_update, xpath_root)

        return update_property(tree_to_update, xpath_root, date_xpaths, prop, values)