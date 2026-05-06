def list_your_courses(self, enrollment_role=None, enrollment_role_id=None, enrollment_state=None, enrollment_type=None, include=None, state=None):
        """
        List your courses.

        Returns the list of active courses for the current user.
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - enrollment_type
        """When set, only return courses where the user is enrolled as this type. For
        example, set to "teacher" to return only courses where the user is
        enrolled as a Teacher.  This argument is ignored if enrollment_role is given."""
        if enrollment_type is not None:
            self._validate_enum(enrollment_type, ["teacher", "student", "ta", "observer", "designer"])
            params["enrollment_type"] = enrollment_type

        # OPTIONAL - enrollment_role
        """Deprecated
        When set, only return courses where the user is enrolled with the specified
        course-level role.  This can be a role created with the
        {api:RoleOverridesController#add_role Add Role API} or a base role type of
        'StudentEnrollment', 'TeacherEnrollment', 'TaEnrollment', 'ObserverEnrollment',
        or 'DesignerEnrollment'."""
        if enrollment_role is not None:
            params["enrollment_role"] = enrollment_role

        # OPTIONAL - enrollment_role_id
        """When set, only return courses where the user is enrolled with the specified
        course-level role.  This can be a role created with the
        {api:RoleOverridesController#add_role Add Role API} or a built_in role type of
        'StudentEnrollment', 'TeacherEnrollment', 'TaEnrollment', 'ObserverEnrollment',
        or 'DesignerEnrollment'."""
        if enrollment_role_id is not None:
            params["enrollment_role_id"] = enrollment_role_id

        # OPTIONAL - enrollment_state
        """When set, only return courses where the user has an enrollment with the given state.
        This will respect section/course/term date overrides."""
        if enrollment_state is not None:
            self._validate_enum(enrollment_state, ["active", "invited_or_pending", "completed"])
            params["enrollment_state"] = enrollment_state

        # OPTIONAL - include
        """- "needs_grading_count": Optional information to include with each Course.
          When needs_grading_count is given, and the current user has grading
          rights, the total number of submissions needing grading for all
          assignments is returned.
        - "syllabus_body": Optional information to include with each Course.
          When syllabus_body is given the user-generated html for the course
          syllabus is returned.
        - "public_description": Optional information to include with each Course.
          When public_description is given the user-generated text for the course
          public description is returned.
        - "total_scores": Optional information to include with each Course.
          When total_scores is given, any student enrollments will also
          include the fields 'computed_current_score', 'computed_final_score',
          'computed_current_grade', and 'computed_final_grade' (see Enrollment
          documentation for more information on these fields). This argument
          is ignored if the course is configured to hide final grades.
        - "current_grading_period_scores": Optional information to include with
          each Course. When current_grading_period_scores is given and total_scores
          is given, any student enrollments will also include the fields
          'multiple_grading_periods_enabled',
          'totals_for_all_grading_periods_option', 'current_grading_period_title',
          'current_grading_period_id', current_period_computed_current_score',
          'current_period_computed_final_score',
          'current_period_computed_current_grade', and
          'current_period_computed_final_grade' (see Enrollment documentation for
          more information on these fields). In addition, when this argument is
          passed, the course will have a 'multiple_grading_periods_enabled' attribute
          on it. This argument is ignored if the course is configured to hide final
          grades or if the total_scores argument is not included.
        - "term": Optional information to include with each Course. When
          term is given, the information for the enrollment term for each course
          is returned.
        - "course_progress": Optional information to include with each Course.
          When course_progress is given, each course will include a
          'course_progress' object with the fields: 'requirement_count', an integer
          specifying the total number of requirements in the course,
          'requirement_completed_count', an integer specifying the total number of
          requirements in this course that have been completed, and
          'next_requirement_url', a string url to the next requirement item, and
          'completed_at', the date the course was completed (null if incomplete).
          'next_requirement_url' will be null if all requirements have been
          completed or the current module does not require sequential progress.
          "course_progress" will return an error message if the course is not
          module based or the user is not enrolled as a student in the course.
        - "sections": Section enrollment information to include with each Course.
          Returns an array of hashes containing the section ID (id), section name
          (name), start and end dates (start_at, end_at), as well as the enrollment
          type (enrollment_role, e.g. 'StudentEnrollment').
        - "storage_quota_used_mb": The amount of storage space used by the files in this course
        - "total_students": Optional information to include with each Course.
          Returns an integer for the total amount of active and invited students.
        - "passback_status": Include the grade passback_status
        - "favorites": Optional information to include with each Course.
          Indicates if the user has marked the course as a favorite course.
        - "teachers": Teacher information to include with each Course.
          Returns an array of hashes containing the {api:Users:UserDisplay UserDisplay} information
          for each teacher in the course.
        - "observed_users": Optional information to include with each Course.
          Will include data for observed users if the current user has an
          observer enrollment.
        - "tabs": Optional information to include with each Course.
          Will include the list of tabs configured for each course.  See the
          {api:TabsController#index List available tabs API} for more information."""
        if include is not None:
            self._validate_enum(include, ["needs_grading_count", "syllabus_body", "public_description", "total_scores", "current_grading_period_scores", "term", "course_progress", "sections", "storage_quota_used_mb", "total_students", "passback_status", "favorites", "teachers", "observed_users"])
            params["include"] = include

        # OPTIONAL - state
        """If set, only return courses that are in the given state(s).
        By default, "available" is returned for students and observers, and
        anything except "deleted", for all other enrollment types"""
        if state is not None:
            self._validate_enum(state, ["unpublished", "available", "completed", "deleted"])
            params["state"] = state

        self.logger.debug("GET /api/v1/courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses".format(**path), data=data, params=params, all_pages=True)