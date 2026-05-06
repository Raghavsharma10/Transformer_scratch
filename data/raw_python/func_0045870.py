def get_response(self, question_id):
        """Gets the response for question_id"""
        question_map = self._get_question_map(question_id)  # will raise NotFound()
        return self._get_response_from_question_map(question_map)