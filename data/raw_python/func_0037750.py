def list_assignment_groups(self, course_id, exclude_assignment_submission_types=None, grading_period_id=None, include=None, override_assignment_dates=None, scope_assignments_to_student=None):
        """
        List assignment groups.

        Returns the list of assignment groups for the current context. The returned
        groups are sorted by their position field.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - include
        """Associations to include with the group. "discussion_topic", "all_dates"
        "assignment_visibility" & "submission" are only valid are only valid if "assignments" is also included.
        The "assignment_visibility" option additionally requires that the Differentiated Assignments course feature be turned on."""
        if include is not None:
            self._validate_enum(include, ["assignments", "discussion_topic", "all_dates", "assignment_visibility", "overrides", "submission"])
            params["include"] = include

        # OPTIONAL - exclude_assignment_submission_types
        """If "assignments" are included, those with the specified submission types
        will be excluded from the assignment groups."""
        if exclude_assignment_submission_types is not None:
            self._validate_enum(exclude_assignment_submission_types, ["online_quiz", "discussion_topic", "wiki_page", "external_tool"])
            params["exclude_assignment_submission_types"] = exclude_assignment_submission_types

        # OPTIONAL - override_assignment_dates
        """Apply assignment overrides for each assignment, defaults to true."""
        if override_assignment_dates is not None:
            params["override_assignment_dates"] = override_assignment_dates

        # OPTIONAL - grading_period_id
        """The id of the grading period in which assignment groups are being requested
        (Requires the Multiple Grading Periods feature turned on.)"""
        if grading_period_id is not None:
            params["grading_period_id"] = grading_period_id

        # OPTIONAL - scope_assignments_to_student
        """If true, all assignments returned will apply to the current user in the
        specified grading period. If assignments apply to other students in the
        specified grading period, but not the current user, they will not be
        returned. (Requires the grading_period_id argument and the Multiple Grading
        Periods feature turned on. In addition, the current user must be a student.)"""
        if scope_assignments_to_student is not None:
            params["scope_assignments_to_student"] = scope_assignments_to_student

        self.logger.debug("GET /api/v1/courses/{course_id}/assignment_groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignment_groups".format(**path), data=data, params=params, all_pages=True)