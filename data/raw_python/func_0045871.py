def get_responses(self):
        """Gets list of the latest responses"""
        response_list = []
        for question_map in self._my_map['questions']:
            response_list.append(self._get_response_from_question_map(question_map))
        return ResponseList(response_list)