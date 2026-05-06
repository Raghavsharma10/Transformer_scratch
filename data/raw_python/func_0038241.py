def enroll_user_courses(self, course_id, enrollment_type, enrollment_user_id, enrollment_associated_user_id=None, enrollment_course_section_id=None, enrollment_enrollment_state=None, enrollment_limit_privileges_to_course_section=None, enrollment_notify=None, enrollment_role=None, enrollment_role_id=None, enrollment_self_enrolled=None, enrollment_self_enrollment_code=None):
        """
        Enroll a user.

        Create a new user enrollment for a course or section.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - enrollment[user_id]
        """The ID of the user to be enrolled in the course."""
        data["enrollment[user_id]"] = enrollment_user_id

        # REQUIRED - enrollment[type]
        """Enroll the user as a student, teacher, TA, observer, or designer. If no
        value is given, the type will be inferred by enrollment[role] if supplied,
        otherwise 'StudentEnrollment' will be used."""
        self._validate_enum(enrollment_type, ["StudentEnrollment", "TeacherEnrollment", "TaEnrollment", "ObserverEnrollment", "DesignerEnrollment"])
        data["enrollment[type]"] = enrollment_type

        # OPTIONAL - enrollment[role]
        """Assigns a custom course-level role to the user."""
        if enrollment_role is not None:
            data["enrollment[role]"] = enrollment_role

        # OPTIONAL - enrollment[role_id]
        """Assigns a custom course-level role to the user."""
        if enrollment_role_id is not None:
            data["enrollment[role_id]"] = enrollment_role_id

        # OPTIONAL - enrollment[enrollment_state]
        """If set to 'active,' student will be immediately enrolled in the course.
        Otherwise they will be required to accept a course invitation. Default is
        'invited.'.
        
        If set to 'inactive', student will be listed in the course roster for
        teachers, but will not be able to participate in the course until
        their enrollment is activated."""
        if enrollment_enrollment_state is not None:
            self._validate_enum(enrollment_enrollment_state, ["active", "invited", "inactive"])
            data["enrollment[enrollment_state]"] = enrollment_enrollment_state

        # OPTIONAL - enrollment[course_section_id]
        """The ID of the course section to enroll the student in. If the
        section-specific URL is used, this argument is redundant and will be
        ignored."""
        if enrollment_course_section_id is not None:
            data["enrollment[course_section_id]"] = enrollment_course_section_id

        # OPTIONAL - enrollment[limit_privileges_to_course_section]
        """If set, the enrollment will only allow the user to see and interact with
        users enrolled in the section given by course_section_id.
        * For teachers and TAs, this includes grading privileges.
        * Section-limited students will not see any users (including teachers
          and TAs) not enrolled in their sections.
        * Users may have other enrollments that grant privileges to
          multiple sections in the same course."""
        if enrollment_limit_privileges_to_course_section is not None:
            data["enrollment[limit_privileges_to_course_section]"] = enrollment_limit_privileges_to_course_section

        # OPTIONAL - enrollment[notify]
        """If true, a notification will be sent to the enrolled user.
        Notifications are not sent by default."""
        if enrollment_notify is not None:
            data["enrollment[notify]"] = enrollment_notify

        # OPTIONAL - enrollment[self_enrollment_code]
        """If the current user is not allowed to manage enrollments in this
        course, but the course allows self-enrollment, the user can self-
        enroll as a student in the default section by passing in a valid
        code. When self-enrolling, the user_id must be 'self'. The
        enrollment_state will be set to 'active' and all other arguments
        will be ignored."""
        if enrollment_self_enrollment_code is not None:
            data["enrollment[self_enrollment_code]"] = enrollment_self_enrollment_code

        # OPTIONAL - enrollment[self_enrolled]
        """If true, marks the enrollment as a self-enrollment, which gives
        students the ability to drop the course if desired. Defaults to false."""
        if enrollment_self_enrolled is not None:
            data["enrollment[self_enrolled]"] = enrollment_self_enrolled

        # OPTIONAL - enrollment[associated_user_id]
        """For an observer enrollment, the ID of a student to observe. The
        caller must have +manage_students+ permission in the course.
        This is a one-off operation; to automatically observe all a
        student's enrollments (for example, as a parent), please use
        the {api:UserObserveesController#create User Observees API}."""
        if enrollment_associated_user_id is not None:
            data["enrollment[associated_user_id]"] = enrollment_associated_user_id

        self.logger.debug("POST /api/v1/courses/{course_id}/enrollments with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/enrollments".format(**path), data=data, params=params, single_item=True)