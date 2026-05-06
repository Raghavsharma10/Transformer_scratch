def list_modules(self, course_id, include=None, search_term=None, student_id=None):
        """
        List modules.

        List the modules in a course
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - include
        """- "items": Return module items inline if possible.
          This parameter suggests that Canvas return module items directly
          in the Module object JSON, to avoid having to make separate API
          requests for each module when enumerating modules and items. Canvas
          is free to omit 'items' for any particular module if it deems them
          too numerous to return inline. Callers must be prepared to use the
          {api:ContextModuleItemsApiController#index List Module Items API}
          if items are not returned.
        - "content_details": Requires include['items']. Returns additional
          details with module items specific to their associated content items.
          Includes standard lock information for each item."""
        if include is not None:
            self._validate_enum(include, ["items", "content_details"])
            params["include"] = include

        # OPTIONAL - search_term
        """The partial name of the modules (and module items, if include['items'] is
        specified) to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - student_id
        """Returns module completion information for the student with this id."""
        if student_id is not None:
            params["student_id"] = student_id

        self.logger.debug("GET /api/v1/courses/{course_id}/modules with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/modules".format(**path), data=data, params=params, all_pages=True)