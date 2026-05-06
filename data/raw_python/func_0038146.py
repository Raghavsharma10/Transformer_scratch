def lists_submissions(self, date, course_id, grader_id, assignment_id):
        """
        Lists submissions.

        Gives a nested list of submission versions
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """The id of the contextual course for this API call"""
        path["course_id"] = course_id

        # REQUIRED - PATH - date
        """The date for which you would like to see submissions"""
        path["date"] = date

        # REQUIRED - PATH - grader_id
        """The ID of the grader for which you want to see submissions"""
        path["grader_id"] = grader_id

        # REQUIRED - PATH - assignment_id
        """The ID of the assignment for which you want to see submissions"""
        path["assignment_id"] = assignment_id

        self.logger.debug("GET /api/v1/courses/{course_id}/gradebook_history/{date}/graders/{grader_id}/assignments/{assignment_id}/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/gradebook_history/{date}/graders/{grader_id}/assignments/{assignment_id}/submissions".format(**path), data=data, params=params, all_pages=True)