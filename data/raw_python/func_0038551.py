def retrieve_assignment_overridden_dates_for_quizzes(self, course_id, quiz_assignment_overrides_0_quiz_ids=None):
        """
        Retrieve assignment-overridden dates for quizzes.

        Retrieve the actual due-at, unlock-at, and available-at dates for quizzes
        based on the assignment overrides active for the current API user.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - quiz_assignment_overrides[0][quiz_ids]
        """An array of quiz IDs. If omitted, overrides for all quizzes available to
        the operating user will be returned."""
        if quiz_assignment_overrides_0_quiz_ids is not None:
            params["quiz_assignment_overrides[0][quiz_ids]"] = quiz_assignment_overrides_0_quiz_ids

        self.logger.debug("GET /api/v1/courses/{course_id}/quizzes/assignment_overrides with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/quizzes/assignment_overrides".format(**path), data=data, params=params, single_item=True)