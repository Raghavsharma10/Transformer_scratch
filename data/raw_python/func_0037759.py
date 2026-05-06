def create_new_course(self, account_id, course_allow_student_forum_attachments=None, course_allow_student_wiki_edits=None, course_allow_wiki_comments=None, course_apply_assignment_group_weights=None, course_course_code=None, course_course_format=None, course_end_at=None, course_grading_standard_id=None, course_hide_final_grades=None, course_integration_id=None, course_is_public=None, course_is_public_to_auth_users=None, course_license=None, course_name=None, course_open_enrollment=None, course_public_description=None, course_public_syllabus=None, course_public_syllabus_to_auth=None, course_restrict_enrollments_to_course_dates=None, course_self_enrollment=None, course_sis_course_id=None, course_start_at=None, course_syllabus_body=None, course_term_id=None, course_time_zone=None, enable_sis_reactivation=None, enroll_me=None, offer=None):
        """
        Create a new course.

        Create a new course
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - course[name]
        """The name of the course. If omitted, the course will be named "Unnamed
        Course." """
        if course_name is not None:
            data["course[name]"] = course_name

        # OPTIONAL - course[course_code]
        """The course code for the course."""
        if course_course_code is not None:
            data["course[course_code]"] = course_course_code

        # OPTIONAL - course[start_at]
        """Course start date in ISO8601 format, e.g. 2011-01-01T01:00Z"""
        if course_start_at is not None:
            data["course[start_at]"] = course_start_at

        # OPTIONAL - course[end_at]
        """Course end date in ISO8601 format. e.g. 2011-01-01T01:00Z"""
        if course_end_at is not None:
            data["course[end_at]"] = course_end_at

        # OPTIONAL - course[license]
        """The name of the licensing. Should be one of the following abbreviations
        (a descriptive name is included in parenthesis for reference):
        - 'private' (Private Copyrighted)
        - 'cc_by_nc_nd' (CC Attribution Non-Commercial No Derivatives)
        - 'cc_by_nc_sa' (CC Attribution Non-Commercial Share Alike)
        - 'cc_by_nc' (CC Attribution Non-Commercial)
        - 'cc_by_nd' (CC Attribution No Derivatives)
        - 'cc_by_sa' (CC Attribution Share Alike)
        - 'cc_by' (CC Attribution)
        - 'public_domain' (Public Domain)."""
        if course_license is not None:
            data["course[license]"] = course_license

        # OPTIONAL - course[is_public]
        """Set to true if course is public to both authenticated and unauthenticated users."""
        if course_is_public is not None:
            data["course[is_public]"] = course_is_public

        # OPTIONAL - course[is_public_to_auth_users]
        """Set to true if course is public only to authenticated users."""
        if course_is_public_to_auth_users is not None:
            data["course[is_public_to_auth_users]"] = course_is_public_to_auth_users

        # OPTIONAL - course[public_syllabus]
        """Set to true to make the course syllabus public."""
        if course_public_syllabus is not None:
            data["course[public_syllabus]"] = course_public_syllabus

        # OPTIONAL - course[public_syllabus_to_auth]
        """Set to true to make the course syllabus public for authenticated users."""
        if course_public_syllabus_to_auth is not None:
            data["course[public_syllabus_to_auth]"] = course_public_syllabus_to_auth

        # OPTIONAL - course[public_description]
        """A publicly visible description of the course."""
        if course_public_description is not None:
            data["course[public_description]"] = course_public_description

        # OPTIONAL - course[allow_student_wiki_edits]
        """If true, students will be able to modify the course wiki."""
        if course_allow_student_wiki_edits is not None:
            data["course[allow_student_wiki_edits]"] = course_allow_student_wiki_edits

        # OPTIONAL - course[allow_wiki_comments]
        """If true, course members will be able to comment on wiki pages."""
        if course_allow_wiki_comments is not None:
            data["course[allow_wiki_comments]"] = course_allow_wiki_comments

        # OPTIONAL - course[allow_student_forum_attachments]
        """If true, students can attach files to forum posts."""
        if course_allow_student_forum_attachments is not None:
            data["course[allow_student_forum_attachments]"] = course_allow_student_forum_attachments

        # OPTIONAL - course[open_enrollment]
        """Set to true if the course is open enrollment."""
        if course_open_enrollment is not None:
            data["course[open_enrollment]"] = course_open_enrollment

        # OPTIONAL - course[self_enrollment]
        """Set to true if the course is self enrollment."""
        if course_self_enrollment is not None:
            data["course[self_enrollment]"] = course_self_enrollment

        # OPTIONAL - course[restrict_enrollments_to_course_dates]
        """Set to true to restrict user enrollments to the start and end dates of the
        course."""
        if course_restrict_enrollments_to_course_dates is not None:
            data["course[restrict_enrollments_to_course_dates]"] = course_restrict_enrollments_to_course_dates

        # OPTIONAL - course[term_id]
        """The unique ID of the term to create to course in."""
        if course_term_id is not None:
            data["course[term_id]"] = course_term_id

        # OPTIONAL - course[sis_course_id]
        """The unique SIS identifier."""
        if course_sis_course_id is not None:
            data["course[sis_course_id]"] = course_sis_course_id

        # OPTIONAL - course[integration_id]
        """The unique Integration identifier."""
        if course_integration_id is not None:
            data["course[integration_id]"] = course_integration_id

        # OPTIONAL - course[hide_final_grades]
        """If this option is set to true, the totals in student grades summary will
        be hidden."""
        if course_hide_final_grades is not None:
            data["course[hide_final_grades]"] = course_hide_final_grades

        # OPTIONAL - course[apply_assignment_group_weights]
        """Set to true to weight final grade based on assignment groups percentages."""
        if course_apply_assignment_group_weights is not None:
            data["course[apply_assignment_group_weights]"] = course_apply_assignment_group_weights

        # OPTIONAL - course[time_zone]
        """The time zone for the course. Allowed time zones are
        {http://www.iana.org/time-zones IANA time zones} or friendlier
        {http://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html Ruby on Rails time zones}."""
        if course_time_zone is not None:
            data["course[time_zone]"] = course_time_zone

        # OPTIONAL - offer
        """If this option is set to true, the course will be available to students
        immediately."""
        if offer is not None:
            data["offer"] = offer

        # OPTIONAL - enroll_me
        """Set to true to enroll the current user as the teacher."""
        if enroll_me is not None:
            data["enroll_me"] = enroll_me

        # OPTIONAL - course[syllabus_body]
        """The syllabus body for the course"""
        if course_syllabus_body is not None:
            data["course[syllabus_body]"] = course_syllabus_body

        # OPTIONAL - course[grading_standard_id]
        """The grading standard id to set for the course.  If no value is provided for this argument the current grading_standard will be un-set from this course."""
        if course_grading_standard_id is not None:
            data["course[grading_standard_id]"] = course_grading_standard_id

        # OPTIONAL - course[course_format]
        """Optional. Specifies the format of the course. (Should be 'on_campus', 'online', or 'blended')"""
        if course_course_format is not None:
            data["course[course_format]"] = course_course_format

        # OPTIONAL - enable_sis_reactivation
        """When true, will first try to re-activate a deleted course with matching sis_course_id if possible."""
        if enable_sis_reactivation is not None:
            data["enable_sis_reactivation"] = enable_sis_reactivation

        self.logger.debug("POST /api/v1/accounts/{account_id}/courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/courses".format(**path), data=data, params=params, single_item=True)