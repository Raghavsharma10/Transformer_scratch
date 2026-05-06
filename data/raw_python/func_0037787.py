def list_submissions_for_multiple_assignments_courses(self, course_id, assignment_ids=None, grading_period_id=None, grouped=None, include=None, order=None, order_direction=None, student_ids=None):
        """
        List submissions for multiple assignments.

        Get all existing submissions for a given set of students and assignments.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - student_ids
        """List of student ids to return submissions for. If this argument is
        omitted, return submissions for the calling user. Students may only list
        their own submissions. Observers may only list those of associated
        students. The special id "all" will return submissions for all students
        in the course/section as appropriate."""
        if student_ids is not None:
            params["student_ids"] = student_ids

        # OPTIONAL - assignment_ids
        """List of assignments to return submissions for. If none are given,
        submissions for all assignments are returned."""
        if assignment_ids is not None:
            params["assignment_ids"] = assignment_ids

        # OPTIONAL - grouped
        """If this argument is present, the response will be grouped by student,
        rather than a flat array of submissions."""
        if grouped is not None:
            params["grouped"] = grouped

        # OPTIONAL - grading_period_id
        """The id of the grading period in which submissions are being requested
        (Requires the Multiple Grading Periods account feature turned on)"""
        if grading_period_id is not None:
            params["grading_period_id"] = grading_period_id

        # OPTIONAL - order
        """The order submissions will be returned in.  Defaults to "id".  Doesn't
        affect results for "grouped" mode."""
        if order is not None:
            self._validate_enum(order, ["id", "graded_at"])
            params["order"] = order

        # OPTIONAL - order_direction
        """Determines whether ordered results are retured in ascending or descending
        order.  Defaults to "ascending".  Doesn't affect results for "grouped" mode."""
        if order_direction is not None:
            self._validate_enum(order_direction, ["ascending", "descending"])
            params["order_direction"] = order_direction

        # OPTIONAL - include
        """Associations to include with the group. `total_scores` requires the
        `grouped` argument."""
        if include is not None:
            self._validate_enum(include, ["submission_history", "submission_comments", "rubric_assessment", "assignment", "total_scores", "visibility", "course", "user"])
            params["include"] = include

        self.logger.debug("GET /api/v1/courses/{course_id}/students/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/students/submissions".format(**path), data=data, params=params, no_data=True)