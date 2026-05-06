def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidRelationshipForm._init_metadata(self, **kwargs)
        update_display_text_defaults(self._mdata['text'], self._locale_map)
        self._text_default = dict(self._mdata['text']['default_string_values'][0])
        self._rating_default = self._mdata['rating']['default_id_values'][0]