def create_assignment_override(self, course_id, assignment_id, assignment_override_course_section_id=None, assignment_override_due_at=None, assignment_override_group_id=None, assignment_override_lock_at=None, assignment_override_student_ids=None, assignment_override_title=None, assignment_override_unlock_at=None):
        """
        Create an assignment override.

        One of student_ids, group_id, or course_section_id must be present. At most
        one should be present; if multiple are present only the most specific
        (student_ids first, then group_id, then course_section_id) is used and any
        others are ignored.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # OPTIONAL - assignment_override[student_ids]
        """The IDs of
        the override's target students. If present, the IDs must each identify a
        user with an active student enrollment in the course that is not already
        targetted by a different adhoc override."""
        if assignment_override_student_ids is not None:
            data["assignment_override[student_ids]"] = assignment_override_student_ids

        # OPTIONAL - assignment_override[title]
        """The title of the adhoc
        assignment override. Required if student_ids is present, ignored
        otherwise (the title is set to the name of the targetted group or section
        instead)."""
        if assignment_override_title is not None:
            data["assignment_override[title]"] = assignment_override_title

        # OPTIONAL - assignment_override[group_id]
        """The ID of the
        override's target group. If present, the following conditions must be met
        for the override to be successful:
        
        1. the assignment MUST be a group assignment (a group_category_id is assigned to it)
        2. the ID must identify an active group in the group set the assignment is in
        3. the ID must not be targetted by a different override
        
        See {Appendix: Group assignments} for more info."""
        if assignment_override_group_id is not None:
            data["assignment_override[group_id]"] = assignment_override_group_id

        # OPTIONAL - assignment_override[course_section_id]
        """The ID
        of the override's target section. If present, must identify an active
        section of the assignment's course not already targetted by a different
        override."""
        if assignment_override_course_section_id is not None:
            data["assignment_override[course_section_id]"] = assignment_override_course_section_id

        # OPTIONAL - assignment_override[due_at]
        """The day/time
        the overridden assignment is due. Accepts times in ISO 8601 format, e.g.
        2014-10-21T18:48:00Z. If absent, this override will not affect due date.
        May be present but null to indicate the override removes any previous due
        date."""
        if assignment_override_due_at is not None:
            data["assignment_override[due_at]"] = assignment_override_due_at

        # OPTIONAL - assignment_override[unlock_at]
        """The day/time
        the overridden assignment becomes unlocked. Accepts times in ISO 8601
        format, e.g. 2014-10-21T18:48:00Z. If absent, this override will not
        affect the unlock date. May be present but null to indicate the override
        removes any previous unlock date."""
        if assignment_override_unlock_at is not None:
            data["assignment_override[unlock_at]"] = assignment_override_unlock_at

        # OPTIONAL - assignment_override[lock_at]
        """The day/time
        the overridden assignment becomes locked. Accepts times in ISO 8601
        format, e.g. 2014-10-21T18:48:00Z. If absent, this override will not
        affect the lock date. May be present but null to indicate the override
        removes any previous lock date."""
        if assignment_override_lock_at is not None:
            data["assignment_override[lock_at]"] = assignment_override_lock_at

        self.logger.debug("POST /api/v1/courses/{course_id}/assignments/{assignment_id}/overrides with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides".format(**path), data=data, params=params, single_item=True)