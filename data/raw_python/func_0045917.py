def get_distribute_alterations_metadata(self):
        """Gets the metadata for the distribute alterations rights flag.

        return: (osid.Metadata) - metadata for the distribution rights
                fields
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['distribute_alterations'])
        metadata.update({'existing_boolean_values': self._my_map['distributeAlterations']})
        return Metadata(**metadata)