def get_public_domain_metadata(self):
        """Gets the metadata for the public domain flag.

        return: (osid.Metadata) - metadata for the public domain
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['public_domain'])
        metadata.update({'existing_boolean_values': self._my_map['publicDomain']})
        return Metadata(**metadata)