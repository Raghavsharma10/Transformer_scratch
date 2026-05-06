def _parse_dates(self, prop=DATES):
        """ Creates and returns a Date Types data structure parsed from the metadata """

        return parse_dates(self._xml_tree, self._data_structures[prop])