def make_account_admin(self, user_id, account_id, role=None, role_id=None, send_confirmation=None):
        """
        Make an account admin.

        Flag an existing user as an admin within the account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - user_id
        """The id of the user to promote."""
        data["user_id"] = user_id

        # OPTIONAL - role
        """(deprecated)
        The user's admin relationship with the account will be created with the
        given role. Defaults to 'AccountAdmin'."""
        if role is not None:
            data["role"] = role

        # OPTIONAL - role_id
        """The user's admin relationship with the account will be created with the
        given role. Defaults to the built-in role for 'AccountAdmin'."""
        if role_id is not None:
            data["role_id"] = role_id

        # OPTIONAL - send_confirmation
        """Send a notification email to
        the new admin if true. Default is true."""
        if send_confirmation is not None:
            data["send_confirmation"] = send_confirmation

        self.logger.debug("POST /api/v1/accounts/{account_id}/admins with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/admins".format(**path), data=data, params=params, single_item=True)