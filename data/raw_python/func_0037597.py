def get_all_quiz_submissions(self, quiz_id, course_id, include=None):
        """
        Get all quiz submissions.

        Get a list of all submissions for this quiz. Users who can view or manage
        grades for a course will have submissions from multiple users returned. A
        user who can only submit will have only their own submissions returned. When
        a user has an in-progress submission, only that submission is returned. When
        there isn't an in-progress quiz_submission, all completed submissions,
        including previous attempts, are returned.
        
        <b>200 OK</b> response code is returned if the request was successful.
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

        # OPTIONAL - include
        """Associations to include with the quiz submission."""
        if include is not None:
            self._validate_enum(include, ["submission", "quiz", "user"])
            params["include"] = include

        self.logger.debug("GET /api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions".format(**path), data=data, params=params, no_data=True)