def get_description_metadata(self):
        """Overrides get_description_metadata of extended object"""
        metadata = dict(self.my_osid_object_form._description_metadata)
        metadata.update({'read_only': True})
        return Metadata(**metadata)