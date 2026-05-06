def delete_external_feed_groups(self, group_id, external_feed_id):
        """
        Delete an external feed.

        Deletes the external feed.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - external_feed_id
        """ID"""
        path["external_feed_id"] = external_feed_id

        self.logger.debug("DELETE /api/v1/groups/{group_id}/external_feeds/{external_feed_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/groups/{group_id}/external_feeds/{external_feed_id}".format(**path), data=data, params=params, single_item=True)