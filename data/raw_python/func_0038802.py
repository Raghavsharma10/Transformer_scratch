def batch_create_overrides_in_course(self, course_id, assignment_overrides):
        """
        Batch create overrides in a course.

        Creates the specified overrides for each assignment.  Handles creation in a
        transaction, so all records are created or none are.
        
        One of student_ids, group_id, or course_section_id must be present. At most
        one should be present; if multiple are present only the most specific
        (student_ids first, then group_id, then course_section_id) is used and any
        others are ignored.
        
        Errors are reported in an errors attribute, an array of errors corresponding
        to inputs.  Global errors will be reported as a single element errors array
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - assignment_overrides
        """Attributes for the new assignment overrides.
        See {api:AssignmentOverridesController#create Create an assignment override} for available
        attributes"""
        data["assignment_overrides"] = assignment_overrides

        self.logger.debug("POST /api/v1/courses/{course_id}/assignments/overrides with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/assignments/overrides".format(**path), data=data, params=params, all_pages=True)