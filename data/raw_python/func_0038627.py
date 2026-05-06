def get_single_external_tool_courses(self, course_id, external_tool_id):
        """
        Get a single external tool.

        Returns the specified external tool.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - external_tool_id
        """ID"""
        path["external_tool_id"] = external_tool_id

        self.logger.debug("GET /api/v1/courses/{course_id}/external_tools/{external_tool_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/external_tools/{external_tool_id}".format(**path), data=data, params=params, no_data=True)