def reorder_pinned_topics_groups(self, order, group_id):
        """
        Reorder pinned topics.

        Puts the pinned discussion topics in the specified order.
        All pinned topics should be included.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - order
        """The ids of the pinned discussion topics in the desired order.
        (For example, "order=104,102,103".)"""
        data["order"] = order

        self.logger.debug("POST /api/v1/groups/{group_id}/discussion_topics/reorder with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/groups/{group_id}/discussion_topics/reorder".format(**path), data=data, params=params, no_data=True)