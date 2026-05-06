def get_display_name_metadata(self):
        """Overrides get_display_name_metadata of extended object"""
        metadata = dict(self.my_osid_object_form._mdata['display_name'])
        metadata.update({'read_only': True})
        return Metadata(**metadata)