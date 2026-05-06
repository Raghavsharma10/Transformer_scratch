def get_single_submission_sections(self, user_id, section_id, assignment_id, include=None):
        """
        Get a single submission.

        Get a single submission, based on user id.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - section_id
        """ID"""
        path["section_id"] = section_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # OPTIONAL - include
        """Associations to include with the group."""
        if include is not None:
            self._validate_enum(include, ["submission_history", "submission_comments", "rubric_assessment", "visibility", "course", "user"])
            params["include"] = include

        self.logger.debug("GET /api/v1/sections/{section_id}/assignments/{assignment_id}/submissions/{user_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/sections/{section_id}/assignments/{assignment_id}/submissions/{user_id}".format(**path), data=data, params=params, no_data=True)