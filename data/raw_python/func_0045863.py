def _get_question_map(self, question_id):
        """get question map from questions matching question_id

        This can make sense of both Section assigned Ids or normal Question/Item Ids

        """
        if question_id.get_authority() == ASSESSMENT_AUTHORITY:
            key = '_id'
            match_value = ObjectId(question_id.get_identifier())
        else:
            key = 'questionId'
            match_value = str(question_id)
        for question_map in self._my_map['questions']:
            if question_map[key] == match_value:
                return question_map
        raise errors.NotFound()