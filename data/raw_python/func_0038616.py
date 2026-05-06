def create_quiz_report(self, quiz_id, course_id, quiz_report_report_type, include=None, quiz_report_includes_all_versions=None):
        """
        Create a quiz report.

        Create and return a new report for this quiz. If a previously
        generated report matches the arguments and is still current (i.e.
        there have been no new submissions), it will be returned.
        
        *Responses*
        
        * <code>400 Bad Request</code> if the specified report type is invalid
        * <code>409 Conflict</code> if a quiz report of the specified type is already being
          generated
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - quiz_id
        """ID"""
        path["quiz_id"] = quiz_id

        # REQUIRED - quiz_report[report_type]
        """The type of report to be generated."""
        self._validate_enum(quiz_report_report_type, ["student_analysis", "item_analysis"])
        data["quiz_report[report_type]"] = quiz_report_report_type

        # OPTIONAL - quiz_report[includes_all_versions]
        """Whether the report should consider all submissions or only the most
        recent. Defaults to false, ignored for item_analysis."""
        if quiz_report_includes_all_versions is not None:
            data["quiz_report[includes_all_versions]"] = quiz_report_includes_all_versions

        # OPTIONAL - include
        """Whether the output should include documents for the file and/or progress
        objects associated with this report. (Note: JSON-API only)"""
        if include is not None:
            self._validate_enum(include, ["file", "progress"])
            data["include"] = include

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/reports with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/reports".format(**path), data=data, params=params, single_item=True)