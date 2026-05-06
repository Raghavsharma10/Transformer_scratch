def get_branding_metadata(self):
        """Gets the metadata for the asset branding.

        return: (osid.Metadata) - metadata for the asset branding.
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._branding_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['brandingIds']})
        return Metadata(**metadata)