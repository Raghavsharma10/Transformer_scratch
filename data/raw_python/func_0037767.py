def permissions(self, course_id, permissions=None):
        """
        Permissions.

        Returns permission information for provided course & current_user
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - permissions
        """List of permissions to check against authenticated user"""
        if permissions is not None:
            params["permissions"] = permissions

        self.logger.debug("GET /api/v1/courses/{course_id}/permissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/permissions".format(**path), data=data, params=params, no_data=True)