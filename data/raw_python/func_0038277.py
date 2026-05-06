def set_extensions_for_student_quiz_submissions(self, user_id, course_id, extend_from_end_at=None, extend_from_now=None, extra_attempts=None, extra_time=None, manually_unlocked=None):
        """
        Set extensions for student quiz submissions.

        <b>Responses</b>
        
        * <b>200 OK</b> if the request was successful
        * <b>403 Forbidden</b> if you are not allowed to extend quizzes for this course
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - user_id
        """The ID of the user we want to add quiz extensions for."""
        data["user_id"] = user_id

        # OPTIONAL - extra_attempts
        """Number of times the student is allowed to re-take the quiz over the
        multiple-attempt limit. This is limited to 1000 attempts or less."""
        if extra_attempts is not None:
            data["extra_attempts"] = extra_attempts

        # OPTIONAL - extra_time
        """The number of extra minutes to allow for all attempts. This will
        add to the existing time limit on the submission. This is limited to
        10080 minutes (1 week)"""
        if extra_time is not None:
            data["extra_time"] = extra_time

        # OPTIONAL - manually_unlocked
        """Allow the student to take the quiz even if it's locked for
        everyone else."""
        if manually_unlocked is not None:
            data["manually_unlocked"] = manually_unlocked

        # OPTIONAL - extend_from_now
        """The number of minutes to extend the quiz from the current time. This is
        mutually exclusive to extend_from_end_at. This is limited to 1440
        minutes (24 hours)"""
        if extend_from_now is not None:
            data["extend_from_now"] = extend_from_now

        # OPTIONAL - extend_from_end_at
        """The number of minutes to extend the quiz beyond the quiz's current
        ending time. This is mutually exclusive to extend_from_now. This is
        limited to 1440 minutes (24 hours)"""
        if extend_from_end_at is not None:
            data["extend_from_end_at"] = extend_from_end_at

        self.logger.debug("POST /api/v1/courses/{course_id}/quiz_extensions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quiz_extensions".format(**path), data=data, params=params, no_data=True)