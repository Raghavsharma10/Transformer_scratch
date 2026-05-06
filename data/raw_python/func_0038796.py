def list_assignment_overrides(self, course_id, assignment_id):
        """
        List assignment overrides.

        Returns the list of overrides for this assignment that target
        sections/groups/students visible to the current user.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        self.logger.debug("GET /api/v1/courses/{course_id}/assignments/{assignment_id}/overrides with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides".format(**path), data=data, params=params, all_pages=True)