def get_question_ids_for_assessment_part(self, assessment_part_id):
        """convenience method returns unique question ids associated with an assessment_part_id"""
        question_ids = []
        for question_map in self._my_map['questions']:
            if question_map['assessmentPartId'] == str(assessment_part_id):
                question_ids.append(self.get_question(question_map=question_map).get_id())
        return question_ids