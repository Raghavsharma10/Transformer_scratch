def get_file_courses(self, id, course_id, include=None):
        """
        Get file.

        Returns the standard attachment json object
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

        # OPTIONAL - include
        """Array of additional information to include.
        
        "user":: the user who uploaded the file or last edited its content
        "usage_rights":: copyright and license information for the file (see UsageRights)"""
        if include is not None:
            self._validate_enum(include, ["user"])
            params["include"] = include

        self.logger.debug("GET /api/v1/courses/{course_id}/files/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/files/{id}".format(**path), data=data, params=params, single_item=True)