def get_genus_type_metadata(self):
        """Overrides get_genus_type_metadata of extended object"""
        metadata = dict(self.my_osid_object_form._genus_type_metadata)
        metadata.update({'read_only': True})
        return Metadata(**metadata)