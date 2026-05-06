def get_learning_objectives_metadata(self):
        """Gets the metadata for learning objectives.

        return: (osid.Metadata) - metadata for the learning objectives
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.get_assets_metadata_template
        metadata = dict(self._mdata['learning_objectives'])
        metadata.update({'existing_learning_objectives_values': self._my_map['learningObjectiveIds']})
        return Metadata(**metadata)