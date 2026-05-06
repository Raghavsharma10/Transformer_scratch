def list_questions_in_quiz_or_submission(self, quiz_id, course_id, quiz_submission_attempt=None, quiz_submission_id=None):
        """
        List questions in a quiz or a submission.

        Returns the list of QuizQuestions in this quiz.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - quiz_id
        """ID"""
        path["quiz_id"] = quiz_id

        # OPTIONAL - quiz_submission_id
        """If specified, the endpoint will return the questions that were presented
        for that submission. This is useful if the quiz has been modified after
        the submission was created and the latest quiz version's set of questions
        does not match the submission's.
        NOTE: you must specify quiz_submission_attempt as well if you specify this
        parameter."""
        if quiz_submission_id is not None:
            params["quiz_submission_id"] = quiz_submission_id

        # OPTIONAL - quiz_submission_attempt
        """The attempt of the submission you want the questions for."""
        if quiz_submission_attempt is not None:
            params["quiz_submission_attempt"] = quiz_submission_attempt

        self.logger.debug("GET /api/v1/courses/{course_id}/quizzes/{quiz_id}/questions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions".format(**path), data=data, params=params, all_pages=True)