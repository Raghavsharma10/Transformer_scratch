def update_assignment_override(self, id, course_id, assignment_id, assignment_override_due_at=None, assignment_override_lock_at=None, assignment_override_student_ids=None, assignment_override_title=None, assignment_override_unlock_at=None):
        """
        Update an assignment override.

        All current overridden values must be supplied if they are to be retained;
        e.g. if due_at was overridden, but this PUT omits a value for due_at,
        due_at will no longer be overridden. If the override is adhoc and
        student_ids is not supplied, the target override set is unchanged. Target
        override sets cannot be changed for group or section overrides.
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

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - assignment_override[student_ids]
        """The IDs of the
        override's target students. If present, the IDs must each identify a
        user with an active student enrollment in the course that is not already
        targetted by a different adhoc override. Ignored unless the override
        being updated is adhoc."""
        if assignment_override_student_ids is not None:
            data["assignment_override[student_ids]"] = assignment_override_student_ids

        # OPTIONAL - assignment_override[title]
        """The title of an adhoc
        assignment override. Ignored unless the override being updated is adhoc."""
        if assignment_override_title is not None:
            data["assignment_override[title]"] = assignment_override_title

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

        self.logger.debug("PUT /api/v1/courses/{course_id}/assignments/{assignment_id}/overrides/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides/{id}".format(**path), data=data, params=params, single_item=True)