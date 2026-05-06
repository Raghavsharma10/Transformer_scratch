def update_module_item(self, id, course_id, module_id, module_item_completion_requirement_min_score=None, module_item_completion_requirement_type=None, module_item_external_url=None, module_item_indent=None, module_item_module_id=None, module_item_new_tab=None, module_item_position=None, module_item_published=None, module_item_title=None):
        """
        Update a module item.

        Update and return an existing module item
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - module_id
        """ID"""
        path["module_id"] = module_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - module_item[title]
        """The name of the module item"""
        if module_item_title is not None:
            data["module_item[title]"] = module_item_title

        # OPTIONAL - module_item[position]
        """The position of this item in the module (1-based)"""
        if module_item_position is not None:
            data["module_item[position]"] = module_item_position

        # OPTIONAL - module_item[indent]
        """0-based indent level; module items may be indented to show a hierarchy"""
        if module_item_indent is not None:
            data["module_item[indent]"] = module_item_indent

        # OPTIONAL - module_item[external_url]
        """External url that the item points to. Only applies to 'ExternalUrl' type."""
        if module_item_external_url is not None:
            data["module_item[external_url]"] = module_item_external_url

        # OPTIONAL - module_item[new_tab]
        """Whether the external tool opens in a new tab. Only applies to
        'ExternalTool' type."""
        if module_item_new_tab is not None:
            data["module_item[new_tab]"] = module_item_new_tab

        # OPTIONAL - module_item[completion_requirement][type]
        """Completion requirement for this module item.
        "must_view": Applies to all item types
        "must_contribute": Only applies to "Assignment", "Discussion", and "Page" types
        "must_submit", "min_score": Only apply to "Assignment" and "Quiz" types
        Inapplicable types will be ignored"""
        if module_item_completion_requirement_type is not None:
            self._validate_enum(module_item_completion_requirement_type, ["must_view", "must_contribute", "must_submit"])
            data["module_item[completion_requirement][type]"] = module_item_completion_requirement_type

        # OPTIONAL - module_item[completion_requirement][min_score]
        """Minimum score required to complete, Required for completion_requirement
        type 'min_score'."""
        if module_item_completion_requirement_min_score is not None:
            data["module_item[completion_requirement][min_score]"] = module_item_completion_requirement_min_score

        # OPTIONAL - module_item[published]
        """Whether the module item is published and visible to students."""
        if module_item_published is not None:
            data["module_item[published]"] = module_item_published

        # OPTIONAL - module_item[module_id]
        """Move this item to another module by specifying the target module id here.
        The target module must be in the same course."""
        if module_item_module_id is not None:
            data["module_item[module_id]"] = module_item_module_id

        self.logger.debug("PUT /api/v1/courses/{course_id}/modules/{module_id}/items/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/modules/{module_id}/items/{id}".format(**path), data=data, params=params, single_item=True)