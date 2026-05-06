def _delete(self):
        """Deletes this AssessmentSection from database.

        Will be called by AssessmentTaken._delete() for clean-up purposes.

        """
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentSection',
                                         runtime=self._runtime)
        collection.delete_one({'_id': ObjectId(self.get_id().get_identifier())})