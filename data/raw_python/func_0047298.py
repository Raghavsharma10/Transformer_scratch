def get_maximum_score_metadata(self):
        """Gets the metadata for the maximum score.

        return: (osid.Metadata) - metadata for the maximum score
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['maximum_score'])
        metadata.update({'existing_cardinal_values': self._my_map['maximumScore']})
        return Metadata(**metadata)