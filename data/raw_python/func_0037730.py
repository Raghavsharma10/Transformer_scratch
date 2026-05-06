def update_user_settings(self, id, collapse_global_nav=None, manual_mark_as_read=None):
        """
        Update user settings.

        Update an existing user's settings.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - manual_mark_as_read
        """If true, require user to manually mark discussion posts as read (don't
        auto-mark as read)."""
        if manual_mark_as_read is not None:
            params["manual_mark_as_read"] = manual_mark_as_read

        # OPTIONAL - collapse_global_nav
        """If true, the user's page loads with the global navigation collapsed"""
        if collapse_global_nav is not None:
            params["collapse_global_nav"] = collapse_global_nav

        self.logger.debug("GET /api/v1/users/{id}/settings with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{id}/settings".format(**path), data=data, params=params, no_data=True)