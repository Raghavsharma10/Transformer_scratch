def create_custom_gradebook_column(self, course_id, column_title, column_hidden=None, column_position=None, column_teacher_notes=None):
        """
        Create a custom gradebook column.

        Create a custom gradebook column
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - column[title]
        """no description"""
        data["column[title]"] = column_title

        # OPTIONAL - column[position]
        """The position of the column relative to other custom columns"""
        if column_position is not None:
            data["column[position]"] = column_position

        # OPTIONAL - column[hidden]
        """Hidden columns are not displayed in the gradebook"""
        if column_hidden is not None:
            data["column[hidden]"] = column_hidden

        # OPTIONAL - column[teacher_notes]
        """Set this if the column is created by a teacher.  The gradebook only
        supports one teacher_notes column."""
        if column_teacher_notes is not None:
            data["column[teacher_notes]"] = column_teacher_notes

        self.logger.debug("POST /api/v1/courses/{course_id}/custom_gradebook_columns with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/custom_gradebook_columns".format(**path), data=data, params=params, single_item=True)