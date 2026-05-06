def get_minimum_score_metadata(self):
        """Gets the metadata for the minimum score.

        return: (osid.Metadata) - metadata for the minimum score
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['minimum_score'])
        metadata.update({'existing_cardinal_values': self._my_map['minimumScore']})
        return Metadata(**metadata)