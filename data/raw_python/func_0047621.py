def get_provenance_id(self):
        """to handle deprecated mecqbank data"""
        if self.has_provenance():
            if 'provenanceId' in self.my_osid_object._my_map:
                return self.my_osid_object._my_map['provenanceId']
            else:
                return self.my_osid_object._my_map['provenanceItemId']
        raise IllegalState()