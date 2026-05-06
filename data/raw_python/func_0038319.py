def create_module_item(self, course_id, module_id, module_item_type, module_item_content_id, module_item_completion_requirement_min_score=None, module_item_completion_requirement_type=None, module_item_external_url=None, module_item_indent=None, module_item_new_tab=None, module_item_page_url=None, module_item_position=None, module_item_title=None):
        """
        Create a module item.

        Create and return a new module item
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

        # OPTIONAL - module_item[title]
        """The name of the module item and associated content"""
        if module_item_title is not None:
            data["module_item[title]"] = module_item_title

        # REQUIRED - module_item[type]
        """The type of content linked to the item"""
        self._validate_enum(module_item_type, ["File", "Page", "Discussion", "Assignment", "Quiz", "SubHeader", "ExternalUrl", "ExternalTool"])
        data["module_item[type]"] = module_item_type

        # REQUIRED - module_item[content_id]
        """The id of the content to link to the module item. Required, except for
        'ExternalUrl', 'Page', and 'SubHeader' types."""
        data["module_item[content_id]"] = module_item_content_id

        # OPTIONAL - module_item[position]
        """The position of this item in the module (1-based)."""
        if module_item_position is not None:
            data["module_item[position]"] = module_item_position

        # OPTIONAL - module_item[indent]
        """0-based indent level; module items may be indented to show a hierarchy"""
        if module_item_indent is not None:
            data["module_item[indent]"] = module_item_indent

        # OPTIONAL - module_item[page_url]
        """Suffix for the linked wiki page (e.g. 'front-page'). Required for 'Page'
        type."""
        if module_item_page_url is not None:
            data["module_item[page_url]"] = module_item_page_url

        # OPTIONAL - module_item[external_url]
        """External url that the item points to. [Required for 'ExternalUrl' and
        'ExternalTool' types."""
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
        """Minimum score required to complete. Required for completion_requirement
        type 'min_score'."""
        if module_item_completion_requirement_min_score is not None:
            data["module_item[completion_requirement][min_score]"] = module_item_completion_requirement_min_score

        self.logger.debug("POST /api/v1/courses/{course_id}/modules/{module_id}/items with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/modules/{module_id}/items".format(**path), data=data, params=params, single_item=True)