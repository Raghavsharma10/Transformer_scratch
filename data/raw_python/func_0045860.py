def _update_from_database(self):
        """Updates map to latest state in database.

        Should be called prior to major object events to assure that an
        assessment being taken on multiple devices are reasonably synchronized.

        """
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentSection',
                                         runtime=self._runtime)
        self._my_map = collection.find_one({'_id': self._my_map['_id']})