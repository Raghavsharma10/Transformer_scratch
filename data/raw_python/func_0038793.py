def list_assignments_for_user(self, user_id, course_id):
        """
        List assignments for user.

        Returns the list of assignments for the specified user if the current user has rights to view.
        See {api:AssignmentsApiController#index List assignments} for valid arguments.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        self.logger.debug("GET /api/v1/users/{user_id}/courses/{course_id}/assignments with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/courses/{course_id}/assignments".format(**path), data=data, params=params, no_data=True)