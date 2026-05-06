def create_quiz_submission_start_quiz_taking_session(self, quiz_id, course_id, access_code=None, preview=None):
        """
        Create the quiz submission (start a quiz-taking session).

        Start taking a Quiz by creating a QuizSubmission which you can use to answer
        questions and submit your answers.
        
        <b>Responses</b>
        
        * <b>200 OK</b> if the request was successful
        * <b>400 Bad Request</b> if the quiz is locked
        * <b>403 Forbidden</b> if an invalid access code is specified
        * <b>403 Forbidden</b> if the Quiz's IP filter restriction does not pass
        * <b>409 Conflict</b> if a QuizSubmission already exists for this user and quiz
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

        # OPTIONAL - access_code
        """Access code for the Quiz, if any."""
        if access_code is not None:
            data["access_code"] = access_code

        # OPTIONAL - preview
        """Whether this should be a preview QuizSubmission and not count towards
        the user's course record. Teachers only."""
        if preview is not None:
            data["preview"] = preview

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions".format(**path), data=data, params=params, no_data=True)