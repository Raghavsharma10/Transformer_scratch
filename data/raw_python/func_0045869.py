def submit_response(self, question_id, answer_form=None):
        """Updates assessmentParts map to insert an item response.

        answer_form is None indicates that the current response is to be cleared

        """
        if answer_form is None:
            response = {'missingResponse': NULL_RESPONSE,
                        'itemId': str(question_id)}
        else:
            response = dict(answer_form._my_map)
            response['submissionTime'] = DateTime.utcnow()
            try:
                response['isCorrect'] = self._get_item(question_id).is_response_correct(
                    Response(osid_object_map=response, runtime=self._runtime, proxy=self._proxy))
            except (errors.IllegalState, errors.NotFound):
                response['isCorrect'] = None
        response['submissionTime'] = DateTime.utcnow()

        question_map = self._get_question_map(question_id)  # will raise NotFound()
        if ('missingResponse' in question_map['responses'][0] and
                question_map['responses'][0]['missingResponse'] == UNANSWERED):
            question_map['responses'] = []  # clear unanswered response
        question_map['responses'].insert(0, response)
        self._save()