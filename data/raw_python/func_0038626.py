def get_sessionless_launch_url_for_external_tool_courses(self, course_id, assignment_id=None, id=None, launch_type=None, module_item_id=None, url=None):
        """
        Get a sessionless launch url for an external tool.

        Returns a sessionless launch url for an external tool.
        
        NOTE: Either the id or url must be provided unless launch_type is assessment or module_item.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - id
        """The external id of the tool to launch."""
        if id is not None:
            params["id"] = id

        # OPTIONAL - url
        """The LTI launch url for the external tool."""
        if url is not None:
            params["url"] = url

        # OPTIONAL - assignment_id
        """The assignment id for an assignment launch. Required if launch_type is set to "assessment"."""
        if assignment_id is not None:
            params["assignment_id"] = assignment_id

        # OPTIONAL - module_item_id
        """The assignment id for a module item launch. Required if launch_type is set to "module_item"."""
        if module_item_id is not None:
            params["module_item_id"] = module_item_id

        # OPTIONAL - launch_type
        """The type of launch to perform on the external tool. Placement names (eg. "course_navigation")
        can also be specified to use the custom launch url for that placement; if done, the tool id
        must be provided."""
        if launch_type is not None:
            self._validate_enum(launch_type, ["assessment", "module_item"])
            params["launch_type"] = launch_type

        self.logger.debug("GET /api/v1/courses/{course_id}/external_tools/sessionless_launch with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/external_tools/sessionless_launch".format(**path), data=data, params=params, no_data=True)