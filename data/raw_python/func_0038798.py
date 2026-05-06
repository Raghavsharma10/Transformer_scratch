def redirect_to_assignment_override_for_section(self, assignment_id, course_section_id):
        """
        Redirect to the assignment override for a section.

        Responds with a redirect to the override for the given section, if any
        (404 otherwise).
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_section_id
        """ID"""
        path["course_section_id"] = course_section_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        self.logger.debug("GET /api/v1/sections/{course_section_id}/assignments/{assignment_id}/override with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/sections/{course_section_id}/assignments/{assignment_id}/override".format(**path), data=data, params=params, no_data=True)