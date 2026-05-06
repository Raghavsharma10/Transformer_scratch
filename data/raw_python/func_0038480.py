def update_single_grading_period(self, id, course_id, grading_periods_end_date, grading_periods_start_date, grading_periods_weight=None):
        """
        Update a single grading period.

        Update an existing grading period.
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

        # REQUIRED - grading_periods[start_date]
        """The date the grading period starts."""
        data["grading_periods[start_date]"] = grading_periods_start_date

        # REQUIRED - grading_periods[end_date]
        """no description"""
        data["grading_periods[end_date]"] = grading_periods_end_date

        # OPTIONAL - grading_periods[weight]
        """A weight value that contributes to the overall weight of a grading period set which is used to calculate how much assignments in this period contribute to the total grade"""
        if grading_periods_weight is not None:
            data["grading_periods[weight]"] = grading_periods_weight

        self.logger.debug("PUT /api/v1/courses/{course_id}/grading_periods/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/grading_periods/{id}".format(**path), data=data, params=params, no_data=True)