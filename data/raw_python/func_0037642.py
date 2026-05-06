def delete_communication_channel_id(self, id, user_id):
        """
        Delete a communication channel.

        Delete an existing communication channel.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("DELETE /api/v1/users/{user_id}/communication_channels/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/users/{user_id}/communication_channels/{id}".format(**path), data=data, params=params, single_item=True)