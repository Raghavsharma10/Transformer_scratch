def _init_map(self):
        """stub"""
        self.my_osid_object_form._my_map['provenanceId'] = \
            self._provenance_metadata['default_object_values'][0]
        if not self.my_osid_object_form.is_for_update():
            if 'effectiveAgentId' in self.my_osid_object_form._kwargs:
                self.my_osid_object_form._my_map['creatorId'] = \
                    str(self.my_osid_object_form._kwargs['effectiveAgentId'])
            else:
                self.my_osid_object_form._my_map['creatorId'] = ''
            self.my_osid_object_form._my_map['creationTime'] = \
                datetime.datetime.now()