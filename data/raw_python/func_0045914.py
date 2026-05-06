def get_distribute_verbatim_metadata(self):
        """Gets the metadata for the distribute verbatim rights flag.

        return: (osid.Metadata) - metadata for the distribution rights
                fields
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['distribute_verbatim'])
        metadata.update({'existing_boolean_values': self._my_map['distributeVerbatim']})
        return Metadata(**metadata)