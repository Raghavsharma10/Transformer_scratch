def export_content_courses(self, course_id, export_type, skip_notifications=None):
        """
        Export content.

        Begin a content export job for a course, group, or user.
        
        You can use the {api:ProgressController#show Progress API} to track the
        progress of the export. The migration's progress is linked to with the
        _progress_url_ value.
        
        When the export completes, use the {api:ContentExportsApiController#show Show content export} endpoint
        to retrieve a download URL for the exported content.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - export_type
        """"common_cartridge":: Export the contents of the course in the Common Cartridge (.imscc) format
        "qti":: Export quizzes from a course in the QTI format
        "zip":: Export files from a course, group, or user in a zip file"""
        self._validate_enum(export_type, ["common_cartridge", "qti", "zip"])
        data["export_type"] = export_type

        # OPTIONAL - skip_notifications
        """Don't send the notifications about the export to the user. Default: false"""
        if skip_notifications is not None:
            data["skip_notifications"] = skip_notifications

        self.logger.debug("POST /api/v1/courses/{course_id}/content_exports with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/content_exports".format(**path), data=data, params=params, single_item=True)