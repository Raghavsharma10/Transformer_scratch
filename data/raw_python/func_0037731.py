def get_custom_color(self, id, asset_string):
        """
        Get custom color.

        Returns the custom colors that have been saved for a user for a given context.
        
        The asset_string parameter should be in the format 'context_id', for example
        'course_42'.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - PATH - asset_string
        """ID"""
        path["asset_string"] = asset_string

        self.logger.debug("GET /api/v1/users/{id}/colors/{asset_string} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{id}/colors/{asset_string}".format(**path), data=data, params=params, no_data=True)