def _save(self):
        """Saves the current state of this AssessmentTaken.

        Should be called every time the sections map changes.

        """
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        collection.save(self._my_map)