def list_external_tools_courses(self, course_id, include_parents=None, search_term=None, selectable=None):
        """
        List external tools.

        Returns the paginated list of external tools for the current context.
        See the get request docs for a single tool for a list of properties on an external tool.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - search_term
        """The partial name of the tools to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - selectable
        """If true, then only tools that are meant to be selectable are returned"""
        if selectable is not None:
            params["selectable"] = selectable

        # OPTIONAL - include_parents
        """If true, then include tools installed in all accounts above the current context"""
        if include_parents is not None:
            params["include_parents"] = include_parents

        self.logger.debug("GET /api/v1/courses/{course_id}/external_tools with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/external_tools".format(**path), data=data, params=params, no_data=True)