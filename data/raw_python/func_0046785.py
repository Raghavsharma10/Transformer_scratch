def get_learning_objective_ids_metadata(self):
        """get the metadata for learning objective"""
        metadata = dict(self._learning_objective_ids_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['learningObjectiveIds'][0]})
        return Metadata(**metadata)