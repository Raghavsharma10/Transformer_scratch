def get_items_sequential_metadata(self):
        """Gets the metadata for sequential operation.

        return: (osid.Metadata) - metadata for the sequential flag
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['items_sequential'])
        metadata.update({'existing_boolean_values': self._my_map['itemsSequential']})
        return Metadata(**metadata)