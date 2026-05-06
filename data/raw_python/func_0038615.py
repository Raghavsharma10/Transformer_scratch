def retrieve_all_quiz_reports(self, quiz_id, course_id, includes_all_versions=None):
        """
        Retrieve all quiz reports.

        Returns a list of all available reports.
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

        # OPTIONAL - includes_all_versions
        """Whether to retrieve reports that consider all the submissions or only
        the most recent. Defaults to false, ignored for item_analysis reports."""
        if includes_all_versions is not None:
            params["includes_all_versions"] = includes_all_versions

        self.logger.debug("GET /api/v1/courses/{course_id}/quizzes/{quiz_id}/reports with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/reports".format(**path), data=data, params=params, all_pages=True)