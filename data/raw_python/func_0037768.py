def copy_course_content(self, course_id, exclude=None, only=None, source_course=None):
        """
        Copy course content.

        DEPRECATED: Please use the {api:ContentMigrationsController#create Content Migrations API}
        
        Copies content from one course into another. The default is to copy all course
        content. You can control specific types to copy by using either the 'except' option
        or the 'only' option.
        
        The response is the same as the course copy status endpoint
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - source_course
        """ID or SIS-ID of the course to copy the content from"""
        if source_course is not None:
            data["source_course"] = source_course

        # OPTIONAL - except
        """A list of the course content types to exclude, all areas not listed will
        be copied."""
        if exclude is not None:
            self._validate_enum(exclude, ["course_settings", "assignments", "external_tools", "files", "topics", "calendar_events", "quizzes", "wiki_pages", "modules", "outcomes"])
            data["except"] = exclude

        # OPTIONAL - only
        """A list of the course content types to copy, all areas not listed will not
        be copied."""
        if only is not None:
            self._validate_enum(only, ["course_settings", "assignments", "external_tools", "files", "topics", "calendar_events", "quizzes", "wiki_pages", "modules", "outcomes"])
            data["only"] = only

        self.logger.debug("POST /api/v1/courses/{course_id}/course_copy with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/course_copy".format(**path), data=data, params=params, no_data=True)