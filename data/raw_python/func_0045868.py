def get_next_question(self, question_id, answered=None, reverse=False, honor_sequential=True):
        """Inspects question map to return the next available question.

        if answered == False: only return next unanswered question
        if answered == True: only return next answered question
        if answered in None: return next question whether answered or not
        if reverse == True: go backwards - effectively get_previous_question
        if honor_sequential == True: only return questions if section or part
                                     is set to sequential items

        """
        self._update_questions()  # Make sure questions list is current
        question_map = self._get_question_map(question_id)  # will raise NotFound()
        questions = list(self._my_map['questions'])
        if reverse:
            questions = questions[::-1]
            error_text = ' previous '
        else:
            if 'missingResponse' in question_map:
                if self._is_question_sequential(question_map) and honor_sequential:
                    raise errors.IllegalState('Next question is not yet available')
            error_text = ' next '
        if questions[-1] == question_map:
            raise errors.IllegalState('No ' + error_text + ' questions available')
        index = questions.index(question_map) + 1
        for question_map in questions[index:]:
            latest_question_response = question_map['responses'][0]
            question_answered = False
            # take missingResponse == UNANSWERED or NULL_RESPONSE as an unanswered question
            if 'missingResponse' not in latest_question_response:
                question_answered = True
            if answered is None or question_answered == answered:
                return self.get_question(question_map=question_map)
        raise errors.IllegalState('No ' + error_text + ' question matching parameters was found')