def _init_data_map(self):
        """ Default data map initialization: MUST be overridden in children """

        if self._data_map is None:
            self._data_map = {'_root': None}
            self._data_map.update({}.fromkeys(self._metadata_props))