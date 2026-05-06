def get_module_item_sequence(self, course_id, asset_id=None, asset_type=None):
        """
        Get module item sequence.

        Given an asset in a course, find the ModuleItem it belongs to, and also the previous and next Module Items
        in the course sequence.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - asset_type
        """The type of asset to find module sequence information for. Use the ModuleItem if it is known
        (e.g., the user navigated from a module item), since this will avoid ambiguity if the asset
        appears more than once in the module sequence."""
        if asset_type is not None:
            self._validate_enum(asset_type, ["ModuleItem", "File", "Page", "Discussion", "Assignment", "Quiz", "ExternalTool"])
            params["asset_type"] = asset_type

        # OPTIONAL - asset_id
        """The id of the asset (or the url in the case of a Page)"""
        if asset_id is not None:
            params["asset_id"] = asset_id

        self.logger.debug("GET /api/v1/courses/{course_id}/module_item_sequence with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/module_item_sequence".format(**path), data=data, params=params, single_item=True)