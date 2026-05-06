def remove_group_from_favorites(self, id):
        """
        Remove group from favorites.

        Remove a group from the current user's favorites.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """the ID or SIS ID of the group to remove"""
        path["id"] = id

        self.logger.debug("DELETE /api/v1/users/self/favorites/groups/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/users/self/favorites/groups/{id}".format(**path), data=data, params=params, single_item=True)