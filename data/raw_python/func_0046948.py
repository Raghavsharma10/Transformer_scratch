def _init_metadata(self, **kwargs):
        """Initialize form metadata"""
        osid_objects.OsidObjectForm._init_metadata(self, **kwargs)
        self._cognitive_process_default = self._mdata['cognitive_process']['default_id_values'][0]
        self._assessment_default = self._mdata['assessment']['default_id_values'][0]
        self._knowledge_category_default = self._mdata['knowledge_category']['default_id_values'][0]