def get_assignment_group(self, course_id, assignment_group_id, grading_period_id=None, include=None, override_assignment_dates=None):
        """
        Get an Assignment Group.

        Returns the assignment group with the given id.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_group_id
        """ID"""
        path["assignment_group_id"] = assignment_group_id

        # OPTIONAL - include
        """Associations to include with the group. "discussion_topic" and "assignment_visibility" and "submission"
        are only valid if "assignments" is also included. The "assignment_visibility" option additionally
        requires that the Differentiated Assignments course feature be turned on."""
        if include is not None:
            self._validate_enum(include, ["assignments", "discussion_topic", "assignment_visibility", "submission"])
            params["include"] = include

        # OPTIONAL - override_assignment_dates
        """Apply assignment overrides for each assignment, defaults to true."""
        if override_assignment_dates is not None:
            params["override_assignment_dates"] = override_assignment_dates

        # OPTIONAL - grading_period_id
        """The id of the grading period in which assignment groups are being requested
        (Requires the Multiple Grading Periods account feature turned on)"""
        if grading_period_id is not None:
            params["grading_period_id"] = grading_period_id

        self.logger.debug("GET /api/v1/courses/{course_id}/assignment_groups/{assignment_group_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignment_groups/{assignment_group_id}".format(**path), data=data, params=params, single_item=True)