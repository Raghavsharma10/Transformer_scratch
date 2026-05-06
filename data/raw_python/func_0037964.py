def add_observee_with_credentials(self, user_id, access_token=None, observee_password=None, observee_unique_id=None):
        """
        Add an observee with credentials.

        Register the given user to observe another user, given the observee's credentials.
        
        *Note:* all users are allowed to add their own observees, given the observee's
        credentials or access token are provided. Administrators can add observees given credentials, access token or
        the {api:UserObserveesController#update observee's id}.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # OPTIONAL - observee[unique_id]
        """The login id for the user to observe.  Required if access_token is omitted."""
        if observee_unique_id is not None:
            data["observee[unique_id]"] = observee_unique_id

        # OPTIONAL - observee[password]
        """The password for the user to observe. Required if access_token is omitted."""
        if observee_password is not None:
            data["observee[password]"] = observee_password

        # OPTIONAL - access_token
        """The access token for the user to observe.  Required if <tt>observee[unique_id]</tt> or <tt>observee[password]</tt> are omitted."""
        if access_token is not None:
            data["access_token"] = access_token

        self.logger.debug("POST /api/v1/users/{user_id}/observees with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/users/{user_id}/observees".format(**path), data=data, params=params, single_item=True)