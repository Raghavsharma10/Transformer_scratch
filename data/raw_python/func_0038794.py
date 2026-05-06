def get_single_assignment(self, id, course_id, all_dates=None, include=None, needs_grading_count_by_section=None, override_assignment_dates=None):
        """
        Get a single assignment.

        Returns the assignment with the given id.
         "observed_users" is passed, submissions for observed users will also be included.
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

        # OPTIONAL - include
        """Associations to include with the assignment. The "assignment_visibility" option
        requires that the Differentiated Assignments course feature be turned on. If"""
        if include is not None:
            self._validate_enum(include, ["submission", "assignment_visibility", "overrides", "observed_users"])
            params["include"] = include

        # OPTIONAL - override_assignment_dates
        """Apply assignment overrides to the assignment, defaults to true."""
        if override_assignment_dates is not None:
            params["override_assignment_dates"] = override_assignment_dates

        # OPTIONAL - needs_grading_count_by_section
        """Split up "needs_grading_count" by sections into the "needs_grading_count_by_section" key, defaults to false"""
        if needs_grading_count_by_section is not None:
            params["needs_grading_count_by_section"] = needs_grading_count_by_section

        # OPTIONAL - all_dates
        """All dates associated with the assignment, if applicable"""
        if all_dates is not None:
            params["all_dates"] = all_dates

        self.logger.debug("GET /api/v1/courses/{course_id}/assignments/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignments/{id}".format(**path), data=data, params=params, single_item=True)