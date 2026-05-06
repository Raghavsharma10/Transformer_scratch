def get_effective_due_dates(self, course_id, assignment_ids=None):
        """
        Get effective due dates.

        For each assignment in the course, returns each assigned student's ID
        and their corresponding due date along with some Multiple Grading Periods
        data. Returns a collection with keys representing assignment IDs and values
        as a collection containing keys representing student IDs and values representing
        the student's effective due_at, the grading_period_id of which the due_at falls
        in, and whether or not the grading period is closed (in_closed_grading_period)
        
        The list of assignment IDs for which effective student due dates are
        requested. If not provided, all assignments in the course will be used.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - assignment_ids
        """no description"""
        if assignment_ids is not None:
            params["assignment_ids"] = assignment_ids

        self.logger.debug("GET /api/v1/courses/{course_id}/effective_due_dates with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/effective_due_dates".format(**path), data=data, params=params, single_item=True)