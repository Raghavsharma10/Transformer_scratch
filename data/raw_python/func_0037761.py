def conclude_course(self, id, event):
        """
        Conclude a course.

        Delete or conclude an existing course
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - event
        """The action to take on the course."""
        self._validate_enum(event, ["delete", "conclude"])
        params["event"] = event

        self.logger.debug("DELETE /api/v1/courses/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/courses/{id}".format(**path), data=data, params=params, no_data=True)