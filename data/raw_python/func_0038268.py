def validate_quiz_access_code(self, id, course_id, access_code):
        """
        Validate quiz access code.

        Accepts an access code and returns a boolean indicating whether that access code is correct
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - access_code
        """The access code being validated"""
        data["access_code"] = access_code

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{id}/validate_access_code with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{id}/validate_access_code".format(**path), data=data, params=params)