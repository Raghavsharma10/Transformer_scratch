def list_assignments(self, course_id, assignment_ids=None, bucket=None, include=None, needs_grading_count_by_section=None, override_assignment_dates=None, search_term=None):
        """
        List assignments.

        Returns the list of assignments for the current context.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - include
        """Associations to include with the assignment. The "assignment_visibility" option
        requires that the Differentiated Assignments course feature be turned on. If
        "observed_users" is passed, submissions for observed users will also be included as an array."""
        if include is not None:
            self._validate_enum(include, ["submission", "assignment_visibility", "all_dates", "overrides", "observed_users"])
            params["include"] = include

        # OPTIONAL - search_term
        """The partial title of the assignments to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - override_assignment_dates
        """Apply assignment overrides for each assignment, defaults to true."""
        if override_assignment_dates is not None:
            params["override_assignment_dates"] = override_assignment_dates

        # OPTIONAL - needs_grading_count_by_section
        """Split up "needs_grading_count" by sections into the "needs_grading_count_by_section" key, defaults to false"""
        if needs_grading_count_by_section is not None:
            params["needs_grading_count_by_section"] = needs_grading_count_by_section

        # OPTIONAL - bucket
        """If included, only return certain assignments depending on due date and submission status."""
        if bucket is not None:
            self._validate_enum(bucket, ["past", "overdue", "undated", "ungraded", "unsubmitted", "upcoming", "future"])
            params["bucket"] = bucket

        # OPTIONAL - assignment_ids
        """if set, return only assignments specified"""
        if assignment_ids is not None:
            params["assignment_ids"] = assignment_ids

        self.logger.debug("GET /api/v1/courses/{course_id}/assignments with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignments".format(**path), data=data, params=params, all_pages=True)