def list_quizzes_in_course(self, course_id, search_term=None):
        """
        List quizzes in a course.

        Returns the list of Quizzes in this course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - search_term
        """The partial title of the quizzes to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        self.logger.debug("GET /api/v1/courses/{course_id}/quizzes with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/quizzes".format(**path), data=data, params=params, all_pages=True)