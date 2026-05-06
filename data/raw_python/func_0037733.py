def merge_user_into_another_user_destination_user_id(self, id, destination_user_id):
        """
        Merge user into another user.

        Merge a user into another user.
        To merge users, the caller must have permissions to manage both users. This
        should be considered irreversible. This will delete the user and move all
        the data into the destination user.
        
        When finding users by SIS ids in different accounts the
        destination_account_id is required.
        
        The account can also be identified by passing the domain in destination_account_id.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - PATH - destination_user_id
        """ID"""
        path["destination_user_id"] = destination_user_id

        self.logger.debug("PUT /api/v1/users/{id}/merge_into/{destination_user_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/{id}/merge_into/{destination_user_id}".format(**path), data=data, params=params, single_item=True)