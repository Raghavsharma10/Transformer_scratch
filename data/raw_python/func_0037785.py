def list_assignment_submissions_courses(self, course_id, assignment_id, grouped=None, include=None):
        """
        List assignment submissions.

        Get all existing submissions for an assignment.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # OPTIONAL - include
        """Associations to include with the group.  "group" will add group_id and group_name."""
        if include is not None:
            self._validate_enum(include, ["submission_history", "submission_comments", "rubric_assessment", "assignment", "visibility", "course", "user", "group"])
            params["include"] = include

        # OPTIONAL - grouped
        """If this argument is true, the response will be grouped by student groups."""
        if grouped is not None:
            params["grouped"] = grouped

        self.logger.debug("GET /api/v1/courses/{course_id}/assignments/{assignment_id}/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions".format(**path), data=data, params=params, all_pages=True)