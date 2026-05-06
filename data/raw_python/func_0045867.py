def get_questions(self, answered=None, honor_sequential=True, update=True):
        """gets all available questions for this section

        if answered == False: only return next unanswered question
        if answered == True: only return next answered question
        if answered in None: return next question whether answered or not
        if honor_sequential == True: only return questions if section or part
                                     is set to sequential items

        """

        def update_question_list():
            """Supportive function to aid readability of _get_questions."""
            latest_question_response = question_map['responses'][0]
            question_answered = False
            # take missingResponse == UNANSWERED or NULL_RESPONSE as an unanswered question
            if 'missingResponse' not in latest_question_response:
                question_answered = True

            if answered is None or answered == question_answered:
                question_list.append(self.get_question(question_map=question_map))
            return question_answered

        prev_question_answered = True
        question_list = []
        if update:
            self._update_questions()  # Make sure questions list is current
        for question_map in self._my_map['questions']:
            if self._is_question_sequential(question_map) and honor_sequential:
                if prev_question_answered:
                    prev_question_answered = update_question_list()
            else:
                update_question_list()
        if self._my_map['actualStartTime'] is None:
            self._my_map['actualStartTime'] = DateTime.utcnow()
        return QuestionList(question_list, runtime=self._runtime, proxy=self._proxy)