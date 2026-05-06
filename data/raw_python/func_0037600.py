def complete_quiz_submission_turn_it_in(self, id, quiz_id, attempt, course_id, validation_token, access_code=None):
        """
        Complete the quiz submission (turn it in).

        Complete the quiz submission by marking it as complete and grading it. When
        the quiz submission has been marked as complete, no further modifications
        will be allowed.
        
        <b>Responses</b>
        
        * <b>200 OK</b> if the request was successful
        * <b>403 Forbidden</b> if an invalid access code is specified
        * <b>403 Forbidden</b> if the Quiz's IP filter restriction does not pass
        * <b>403 Forbidden</b> if an invalid token is specified
        * <b>400 Bad Request</b> if the QS is already complete
        * <b>400 Bad Request</b> if the attempt parameter is missing
        * <b>400 Bad Request</b> if the attempt parameter is not the latest attempt
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

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - attempt
        """The attempt number of the quiz submission that should be completed. Note
        that this must be the latest attempt index, as earlier attempts can not
        be modified."""
        data["attempt"] = attempt

        # REQUIRED - validation_token
        """The unique validation token you received when this Quiz Submission was
        created."""
        data["validation_token"] = validation_token

        # OPTIONAL - access_code
        """Access code for the Quiz, if any."""
        if access_code is not None:
            data["access_code"] = access_code

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{id}/complete with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{id}/complete".format(**path), data=data, params=params, no_data=True)