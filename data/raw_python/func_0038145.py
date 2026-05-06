def details_for_given_date_in_gradebook_history_for_this_course(self, date, course_id):
        """
        Details for a given date in gradebook history for this course.

        Returns the graders who worked on this day, along with the assignments they worked on.
        More details can be obtained by selecting a grader and assignment and calling the
        'submissions' api endpoint for a given date.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """The id of the contextual course for this API call"""
        path["course_id"] = course_id

        # REQUIRED - PATH - date
        """The date for which you would like to see detailed information"""
        path["date"] = date

        self.logger.debug("GET /api/v1/courses/{course_id}/gradebook_history/{date} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/gradebook_history/{date}".format(**path), data=data, params=params, all_pages=True)