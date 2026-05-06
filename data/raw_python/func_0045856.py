def _save(self):
        """Saves the current state of this AssessmentSection to database.

        Should be called every time the question map changes.

        """
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentSection',
                                         runtime=self._runtime)
        if '_id' in self._my_map:  # This is the first time:
            collection.save(self._my_map)
        else:
            insert_result = collection.insert_one(self._my_map)
            self._my_map = collection.find_one({'_id': insert_result.inserted_id})