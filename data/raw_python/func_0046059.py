def set_provenance(self, provenance_id):
        """stub"""
        if not self.my_osid_object_form._is_valid_string(
                provenance_id, self.get_provenance_metadata()):
            raise InvalidArgument('provenanceId')
        self.my_osid_object_form._my_map['provenanceId'] = provenance_id