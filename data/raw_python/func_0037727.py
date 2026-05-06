def list_users_in_account(self, account_id, search_term=None):
        """
        List users in account.

        Retrieve the list of users associated with this account.
        
         @example_request
           curl https://<canvas>/api/v1/accounts/self/users?search_term=<search value> \
              -X GET \
              -H 'Authorization: Bearer <token>'
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - search_term
        """The partial name or full ID of the users to match and return in the
        results list. Must be at least 3 characters.
        
        Note that the API will prefer matching on canonical user ID if the ID has
        a numeric form. It will only search against other fields if non-numeric
        in form, or if the numeric value doesn't yield any matches. Queries by
        administrative users will search on SIS ID, name, or email address; non-
        administrative queries will only be compared against name."""
        if search_term is not None:
            params["search_term"] = search_term

        self.logger.debug("GET /api/v1/accounts/{account_id}/users with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/users".format(**path), data=data, params=params, all_pages=True)