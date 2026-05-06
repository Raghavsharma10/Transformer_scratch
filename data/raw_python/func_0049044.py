def get_rating_metadata(self):
        """Gets the metadata for a rating.

        return: (osid.Metadata) - metadata for the rating
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['rating'])
        metadata.update({'existing_id_values': self._my_map['ratingId']})
        return Metadata(**metadata)