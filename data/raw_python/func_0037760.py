def list_users_in_course_users(self, course_id, enrollment_role=None, enrollment_role_id=None, enrollment_state=None, enrollment_type=None, include=None, search_term=None, user_id=None, user_ids=None):
        """
        List users in course.

        Returns the list of users in this course. And optionally the user's enrollments in the course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - search_term
        """The partial name or full ID of the users to match and return in the results list."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - enrollment_type
        """When set, only return users where the user is enrolled as this type.
        "student_view" implies include[]=test_student.
        This argument is ignored if enrollment_role is given."""
        if enrollment_type is not None:
            self._validate_enum(enrollment_type, ["teacher", "student", "student_view", "ta", "observer", "designer"])
            params["enrollment_type"] = enrollment_type

        # OPTIONAL - enrollment_role
        """Deprecated
        When set, only return users enrolled with the specified course-level role.  This can be
        a role created with the {api:RoleOverridesController#add_role Add Role API} or a
        base role type of 'StudentEnrollment', 'TeacherEnrollment', 'TaEnrollment',
        'ObserverEnrollment', or 'DesignerEnrollment'."""
        if enrollment_role is not None:
            params["enrollment_role"] = enrollment_role

        # OPTIONAL - enrollment_role_id
        """When set, only return courses where the user is enrolled with the specified
        course-level role.  This can be a role created with the
        {api:RoleOverridesController#add_role Add Role API} or a built_in role id with type
        'StudentEnrollment', 'TeacherEnrollment', 'TaEnrollment', 'ObserverEnrollment',
        or 'DesignerEnrollment'."""
        if enrollment_role_id is not None:
            params["enrollment_role_id"] = enrollment_role_id

        # OPTIONAL - include
        """- "email": Optional user email.
        - "enrollments":
        Optionally include with each Course the user's current and invited
        enrollments. If the user is enrolled as a student, and the account has
        permission to manage or view all grades, each enrollment will include a
        'grades' key with 'current_score', 'final_score', 'current_grade' and
        'final_grade' values.
        - "locked": Optionally include whether an enrollment is locked.
        - "avatar_url": Optionally include avatar_url.
        - "bio": Optionally include each user's bio.
        - "test_student": Optionally include the course's Test Student,
        if present. Default is to not include Test Student.
        - "custom_links": Optionally include plugin-supplied custom links for each student,
        such as analytics information"""
        if include is not None:
            self._validate_enum(include, ["email", "enrollments", "locked", "avatar_url", "test_student", "bio", "custom_links"])
            params["include"] = include

        # OPTIONAL - user_id
        """If this parameter is given and it corresponds to a user in the course,
        the +page+ parameter will be ignored and the page containing the specified user
        will be returned instead."""
        if user_id is not None:
            params["user_id"] = user_id

        # OPTIONAL - user_ids
        """If included, the course users set will only include users with IDs
        specified by the param. Note: this will not work in conjunction
        with the "user_id" argument but multiple user_ids can be included."""
        if user_ids is not None:
            params["user_ids"] = user_ids

        # OPTIONAL - enrollment_state
        """When set, only return users where the enrollment workflow state is of one of the given types.
        "active" and "invited" enrollments are returned by default."""
        if enrollment_state is not None:
            self._validate_enum(enrollment_state, ["active", "invited", "rejected", "completed", "inactive"])
            params["enrollment_state"] = enrollment_state

        self.logger.debug("GET /api/v1/courses/{course_id}/users with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/users".format(**path), data=data, params=params, all_pages=True)