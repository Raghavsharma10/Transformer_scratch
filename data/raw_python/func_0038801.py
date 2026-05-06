def batch_retrieve_overrides_in_course(self, course_id, assignment_overrides_id, assignment_overrides_assignment_id):
        """
        Batch retrieve overrides in a course.

        Returns a list of specified overrides in this course, providing
        they target sections/groups/students visible to the current user.
        Returns null elements in the list for requests that were not found.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - assignment_overrides[id]
        """Ids of overrides to retrieve"""
        params["assignment_overrides[id]"] = assignment_overrides_id

        # REQUIRED - assignment_overrides[assignment_id]
        """Ids of assignments for each override"""
        params["assignment_overrides[assignment_id]"] = assignment_overrides_assignment_id

        self.logger.debug("GET /api/v1/courses/{course_id}/assignments/overrides with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/assignments/overrides".format(**path), data=data, params=params, all_pages=True)