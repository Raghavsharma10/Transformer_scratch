def preview_processed_html(self, group_id, html=None):
        """
        Preview processed html.

        Preview html content processed for this group
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # OPTIONAL - html
        """The html content to process"""
        if html is not None:
            data["html"] = html

        self.logger.debug("POST /api/v1/groups/{group_id}/preview_html with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/groups/{group_id}/preview_html".format(**path), data=data, params=params, no_data=True)