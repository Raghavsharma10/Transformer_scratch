def reset_course(self, course_id):
        """
        Reset a course.

        Deletes the current course, and creates a new equivalent course with
        no content, but all sections and users moved over.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        self.logger.debug("POST /api/v1/courses/{course_id}/reset_content with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/reset_content".format(**path), data=data, params=params, single_item=True)