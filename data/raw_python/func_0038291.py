def create_course_section(self, course_id, course_section_end_at=None, course_section_name=None, course_section_restrict_enrollments_to_section_dates=None, course_section_sis_section_id=None, course_section_start_at=None, enable_sis_reactivation=None):
        """
        Create course section.

        Creates a new section for this course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - course_section[name]
        """The name of the section"""
        if course_section_name is not None:
            data["course_section[name]"] = course_section_name

        # OPTIONAL - course_section[sis_section_id]
        """The sis ID of the section"""
        if course_section_sis_section_id is not None:
            data["course_section[sis_section_id]"] = course_section_sis_section_id

        # OPTIONAL - course_section[start_at]
        """Section start date in ISO8601 format, e.g. 2011-01-01T01:00Z"""
        if course_section_start_at is not None:
            data["course_section[start_at]"] = course_section_start_at

        # OPTIONAL - course_section[end_at]
        """Section end date in ISO8601 format. e.g. 2011-01-01T01:00Z"""
        if course_section_end_at is not None:
            data["course_section[end_at]"] = course_section_end_at

        # OPTIONAL - course_section[restrict_enrollments_to_section_dates]
        """Set to true to restrict user enrollments to the start and end dates of the section."""
        if course_section_restrict_enrollments_to_section_dates is not None:
            data["course_section[restrict_enrollments_to_section_dates]"] = course_section_restrict_enrollments_to_section_dates

        # OPTIONAL - enable_sis_reactivation
        """When true, will first try to re-activate a deleted section with matching sis_section_id if possible."""
        if enable_sis_reactivation is not None:
            data["enable_sis_reactivation"] = enable_sis_reactivation

        self.logger.debug("POST /api/v1/courses/{course_id}/sections with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/sections".format(**path), data=data, params=params, single_item=True)