def list_active_courses_in_account(self, account_id, by_subaccounts=None, by_teachers=None, completed=None, enrollment_term_id=None, enrollment_type=None, hide_enrollmentless_courses=None, include=None, published=None, search_term=None, state=None, with_enrollments=None):
        """
        List active courses in an account.

        Retrieve the list of courses in this account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - with_enrollments
        """If true, include only courses with at least one enrollment.  If false,
        include only courses with no enrollments.  If not present, do not filter
        on course enrollment status."""
        if with_enrollments is not None:
            params["with_enrollments"] = with_enrollments

        # OPTIONAL - enrollment_type
        """If set, only return courses that have at least one user enrolled in
        in the course with one of the specified enrollment types."""
        if enrollment_type is not None:
            self._validate_enum(enrollment_type, ["teacher", "student", "ta", "observer", "designer"])
            params["enrollment_type"] = enrollment_type

        # OPTIONAL - published
        """If true, include only published courses.  If false, exclude published
        courses.  If not present, do not filter on published status."""
        if published is not None:
            params["published"] = published

        # OPTIONAL - completed
        """If true, include only completed courses (these may be in state
        'completed', or their enrollment term may have ended).  If false, exclude
        completed courses.  If not present, do not filter on completed status."""
        if completed is not None:
            params["completed"] = completed

        # OPTIONAL - by_teachers
        """List of User IDs of teachers; if supplied, include only courses taught by
        one of the referenced users."""
        if by_teachers is not None:
            params["by_teachers"] = by_teachers

        # OPTIONAL - by_subaccounts
        """List of Account IDs; if supplied, include only courses associated with one
        of the referenced subaccounts."""
        if by_subaccounts is not None:
            params["by_subaccounts"] = by_subaccounts

        # OPTIONAL - hide_enrollmentless_courses
        """If present, only return courses that have at least one enrollment.
        Equivalent to 'with_enrollments=true'; retained for compatibility."""
        if hide_enrollmentless_courses is not None:
            params["hide_enrollmentless_courses"] = hide_enrollmentless_courses

        # OPTIONAL - state
        """If set, only return courses that are in the given state(s). By default,
        all states but "deleted" are returned."""
        if state is not None:
            self._validate_enum(state, ["created", "claimed", "available", "completed", "deleted", "all"])
            params["state"] = state

        # OPTIONAL - enrollment_term_id
        """If set, only includes courses from the specified term."""
        if enrollment_term_id is not None:
            params["enrollment_term_id"] = enrollment_term_id

        # OPTIONAL - search_term
        """The partial course name, code, or full ID to match and return in the results list. Must be at least 3 characters."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - include
        """- All explanations can be seen in the {api:CoursesController#index Course API index documentation}
        - "sections", "needs_grading_count" and "total_scores" are not valid options at the account level"""
        if include is not None:
            self._validate_enum(include, ["syllabus_body", "term", "course_progress", "storage_quota_used_mb", "total_students", "teachers"])
            params["include"] = include

        self.logger.debug("GET /api/v1/accounts/{account_id}/courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/courses".format(**path), data=data, params=params, all_pages=True)