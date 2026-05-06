def update_column_data(self, id, user_id, course_id, column_data_content):
        """
        Update column data.

        Set the content of a custom column
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - column_data[content]
        """Column content.  Setting this to blank will delete the datum object."""
        data["column_data[content]"] = column_data_content

        self.logger.debug("PUT /api/v1/courses/{course_id}/custom_gradebook_columns/{id}/data/{user_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/custom_gradebook_columns/{id}/data/{user_id}".format(**path), data=data, params=params, single_item=True)