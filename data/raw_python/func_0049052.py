def _init_metadata(self):
        """Initialize OsidObjectForm metadata."""

        # pylint: disable=attribute-defined-outside-init
        # this method is called from descendent __init__
        self._mdata.update(default_mdata.get_osid_form_mdata())
        update_display_text_defaults(self._mdata['journal_comment'], self._locale_map)
        for element_name in self._mdata:
            self._mdata[element_name].update(
                {'element_id': Id(self._authority,
                                  self._namespace,
                                  element_name)})
        self._journal_comment_default = self._mdata['journal_comment']['default_string_values'][0]
        self._validation_messages = {}