def select_mastery_path(self, id, course_id, module_id, assignment_set_id=None, student_id=None):
        """
        Select a mastery path.

        Select a mastery path when module item includes several possible paths.
        Requires Mastery Paths feature to be enabled.  Returns a compound document
        with the assignments included in the given path and any module items
        related to those assignments
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - module_id
        """ID"""
        path["module_id"] = module_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - assignment_set_id
        """Assignment set chosen, as specified in the mastery_paths portion of the
        context module item response"""
        if assignment_set_id is not None:
            data["assignment_set_id"] = assignment_set_id

        # OPTIONAL - student_id
        """Which student the selection applies to.  If not specified, current user is
        implied."""
        if student_id is not None:
            data["student_id"] = student_id

        self.logger.debug("POST /api/v1/courses/{course_id}/modules/{module_id}/items/{id}/select_mastery_path with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/modules/{module_id}/items/{id}/select_mastery_path".format(**path), data=data, params=params, no_data=True)