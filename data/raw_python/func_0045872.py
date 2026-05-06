def is_question_answered(self, question_id):
        """has the question matching item_id been answered and not skipped"""
        question_map = self._get_question_map(question_id)  # will raise NotFound()
        if 'missingResponse' in question_map['responses'][0]:
            return False
        else:
            return True