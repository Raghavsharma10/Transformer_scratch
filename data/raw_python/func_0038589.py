def query_by_grader(self, grader_id, end_time=None, start_time=None):
        """
        Query by grader.

        List grade change events for a given grader.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - grader_id
        """ID"""
        path["grader_id"] = grader_id

        # OPTIONAL - start_time
        """The beginning of the time range from which you want events."""
        if start_time is not None:
            params["start_time"] = start_time

        # OPTIONAL - end_time
        """The end of the time range from which you want events."""
        if end_time is not None:
            params["end_time"] = end_time

        self.logger.debug("GET /api/v1/audit/grade_change/graders/{grader_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/audit/grade_change/graders/{grader_id}".format(**path), data=data, params=params, all_pages=True)