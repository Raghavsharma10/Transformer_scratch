def _init_metadata(self, **kwargs):
        """Initialize form metadata"""

        osid_objects.OsidSourceableForm._init_metadata(self)
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._copyright_registration_default = self._mdata['copyright_registration']['default_string_values'][0]
        update_display_text_defaults(self._mdata['copyright'], self._locale_map)
        self._copyright_default = dict(self._mdata['copyright']['default_string_values'][0])
        update_display_text_defaults(self._mdata['title'], self._locale_map)
        self._title_default = dict(self._mdata['title']['default_string_values'][0])
        self._distribute_verbatim_default = self._mdata['distribute_verbatim']['default_boolean_values'][0]
        self._created_date_default = self._mdata['created_date']['default_date_time_values'][0]
        self._distribute_alterations_default = self._mdata['distribute_alterations']['default_boolean_values'][0]
        update_display_text_defaults(self._mdata['principal_credit_string'], self._locale_map)
        self._principal_credit_string_default = dict(self._mdata['principal_credit_string']['default_string_values'][0])
        self._published_date_default = self._mdata['published_date']['default_date_time_values'][0]
        self._source_default = self._mdata['source']['default_id_values'][0]
        self._provider_links_default = self._mdata['provider_links']['default_id_values']
        self._public_domain_default = self._mdata['public_domain']['default_boolean_values'][0]
        self._distribute_compositions_default = self._mdata['distribute_compositions']['default_boolean_values'][0]
        self._composition_default = self._mdata['composition']['default_id_values'][0]
        self._published_default = self._mdata['published']['default_boolean_values'][0]