def update_student_question_scores_and_comments(self, id, quiz_id, attempt, course_id, fudge_points=None, questions=None):
        """
        Update student question scores and comments.

        Update the amount of points a student has scored for questions they've
        answered, provide comments for the student about their answer(s), or simply
        fudge the total score by a specific amount of points.
        
        <b>Responses</b>
        
        * <b>200 OK</b> if the request was successful
        * <b>403 Forbidden</b> if you are not a teacher in this course
        * <b>400 Bad Request</b> if the attempt parameter is missing or invalid
        * <b>400 Bad Request</b> if the specified QS attempt is not yet complete
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
        """The attempt number of the quiz submission that should be updated. This
        attempt MUST be already completed."""
        data["attempt"] = attempt

        # OPTIONAL - fudge_points
        """Amount of positive or negative points to fudge the total score by."""
        if fudge_points is not None:
            data["fudge_points"] = fudge_points

        # OPTIONAL - questions
        """A set of scores and comments for each question answered by the student.
        The keys are the question IDs, and the values are hashes of `score` and
        `comment` entries. See {Appendix: Manual Scoring} for more on this
        parameter."""
        if questions is not None:
            data["questions"] = questions

        self.logger.debug("PUT /api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/submissions/{id}".format(**path), data=data, params=params, no_data=True)