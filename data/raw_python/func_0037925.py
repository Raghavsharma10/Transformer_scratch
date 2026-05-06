def list_files_courses(self, course_id, content_types=None, include=None, only=None, order=None, search_term=None, sort=None):
        """
        List files.

        Returns the paginated list of files for the folder or course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - content_types
        """Filter results by content-type. You can specify type/subtype pairs (e.g.,
        'image/jpeg'), or simply types (e.g., 'image', which will match
        'image/gif', 'image/jpeg', etc.)."""
        if content_types is not None:
            params["content_types"] = content_types

        # OPTIONAL - search_term
        """The partial name of the files to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - include
        """Array of additional information to include.
        
        "user":: the user who uploaded the file or last edited its content
        "usage_rights":: copyright and license information for the file (see UsageRights)"""
        if include is not None:
            self._validate_enum(include, ["user"])
            params["include"] = include

        # OPTIONAL - only
        """Array of information to restrict to. Overrides include[]
        
        "names":: only returns file name information"""
        if only is not None:
            params["only"] = only

        # OPTIONAL - sort
        """Sort results by this field. Defaults to 'name'. Note that `sort=user` implies `include[]=user`."""
        if sort is not None:
            self._validate_enum(sort, ["name", "size", "created_at", "updated_at", "content_type", "user"])
            params["sort"] = sort

        # OPTIONAL - order
        """The sorting order. Defaults to 'asc'."""
        if order is not None:
            self._validate_enum(order, ["asc", "desc"])
            params["order"] = order

        self.logger.debug("GET /api/v1/courses/{course_id}/files with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/files".format(**path), data=data, params=params, all_pages=True)