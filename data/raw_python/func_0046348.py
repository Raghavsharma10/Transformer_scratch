def get_question(self):
        """Gets the question.

        return: (osid.assessment.Question) - the question
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        question_map = dict(self._my_map['question'])
        question_map['learningObjectiveIds'] = self._my_map['learningObjectiveIds']
        return Question(osid_object_map=question_map,
                        runtime=self._runtime,
                        proxy=self._proxy)