def _init_map(self):
        """stub"""
        super(MultiLanguageDragAndDropQuestionFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['droppables'] = \
            self._droppables_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['targets'] = \
            self._targets_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['zones'] = \
            self._zones_metadata['default_object_values'][0]
        self.my_osid_object_form._my_map['shuffleDroppables'] = \
            bool(self._shuffle_droppables_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['shuffleTargets'] = \
            bool(self._shuffle_targets_metadata['default_boolean_values'][0])
        self.my_osid_object_form._my_map['shuffleZones'] = \
            bool(self._shuffle_zones_metadata['default_boolean_values'][0])