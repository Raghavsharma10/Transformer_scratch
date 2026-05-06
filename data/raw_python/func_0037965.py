def show_observee(self, user_id, observee_id):
        """
        Show an observee.

        Gets information about an observed user.
        
        *Note:* all users are allowed to view their own observees.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - PATH - observee_id
        """ID"""
        path["observee_id"] = observee_id

        self.logger.debug("GET /api/v1/users/{user_id}/observees/{observee_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/observees/{observee_id}".format(**path), data=data, params=params, single_item=True)