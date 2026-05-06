def answering_questions(self, attempt, validation_token, quiz_submission_id, access_code=None, quiz_questions=None):
        """
        Answering questions.

        Provide or update an answer to one or more QuizQuestions.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - quiz_submission_id
        """ID"""
        path["quiz_submission_id"] = quiz_submission_id

        # REQUIRED - attempt
        """The attempt number of the quiz submission being taken. Note that this
        must be the latest attempt index, as questions for earlier attempts can
        not be modified."""
        data["attempt"] = attempt

        # REQUIRED - validation_token
        """The unique validation token you received when the Quiz Submission was
        created."""
        data["validation_token"] = validation_token

        # OPTIONAL - access_code
        """Access code for the Quiz, if any."""
        if access_code is not None:
            data["access_code"] = access_code

        # OPTIONAL - quiz_questions
        """Set of question IDs and the answer value.
        
        See {Appendix: Question Answer Formats} for the accepted answer formats
        for each question type."""
        if quiz_questions is not None:
            data["quiz_questions"] = quiz_questions

        self.logger.debug("POST /api/v1/quiz_submissions/{quiz_submission_id}/questions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/quiz_submissions/{quiz_submission_id}/questions".format(**path), data=data, params=params, all_pages=True)