def get_weight_metadata(self):
        """Gets the metadata for the weight.

        return: (osid.Metadata) - metadata for the weight
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['weight'])
        metadata.update({'existing_cardinal_values': self._my_map['weight']})
        return Metadata(**metadata)