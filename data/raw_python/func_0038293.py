def edit_section(self, id, course_section_end_at=None, course_section_name=None, course_section_restrict_enrollments_to_section_dates=None, course_section_sis_section_id=None, course_section_start_at=None):
        """
        Edit a section.

        Modify an existing section.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

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

        self.logger.debug("PUT /api/v1/sections/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/sections/{id}".format(**path), data=data, params=params, single_item=True)