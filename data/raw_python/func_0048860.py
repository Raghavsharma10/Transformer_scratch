def _get_assessment_part_collection(self, assessment_part_id):
        """Returns a Mongo Collection and AssessmentPart given a AssessmentPart Id"""
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        assessment_part_map = collection.find_one({'_id': ObjectId(assessment_part_id.get_identifier())})
        if 'itemIds' not in assessment_part_map:
            raise errors.NotFound('no Items are assigned to this AssessmentPart')
        return assessment_part_map, collection