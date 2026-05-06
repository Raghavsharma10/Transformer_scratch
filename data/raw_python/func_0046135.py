def get_license_metadata(self):
        """Gets the metadata for the license.

        return: (osid.Metadata) - metadata for the license
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._license_metadata)
        metadata.update({'existing_string_values': self.my_osid_object_form._my_map['license']})
        return Metadata(**metadata)