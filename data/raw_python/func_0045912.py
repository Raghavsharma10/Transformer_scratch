def get_copyright_registration_metadata(self):
        """Gets the metadata for the copyright registration.

        return: (osid.Metadata) - metadata for the copyright
                registration
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['copyright_registration'])
        metadata.update({'existing_string_values': self._my_map['copyrightRegistration']})
        return Metadata(**metadata)