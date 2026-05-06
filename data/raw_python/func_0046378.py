def get_items_shuffled_metadata(self):
        """Gets the metadata for shuffling items.

        return: (osid.Metadata) - metadata for the shuffled flag
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['items_shuffled'])
        metadata.update({'existing_boolean_values': self._my_map['itemsShuffled']})
        return Metadata(**metadata)