def edit_user_login(self, id, account_id, login_integration_id=None, login_password=None, login_sis_user_id=None, login_unique_id=None):
        """
        Edit a user login.

        Update an existing login for a user in the given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - login[unique_id]
        """The new unique ID for the login."""
        if login_unique_id is not None:
            data["login[unique_id]"] = login_unique_id

        # OPTIONAL - login[password]
        """The new password for the login. Can only be set by an admin user if admins
        are allowed to change passwords for the account."""
        if login_password is not None:
            data["login[password]"] = login_password

        # OPTIONAL - login[sis_user_id]
        """SIS ID for the login. To set this parameter, the caller must be able to
        manage SIS permissions on the account."""
        if login_sis_user_id is not None:
            data["login[sis_user_id]"] = login_sis_user_id

        # OPTIONAL - login[integration_id]
        """Integration ID for the login. To set this parameter, the caller must be able to
        manage SIS permissions on the account. The Integration ID is a secondary
        identifier useful for more complex SIS integrations."""
        if login_integration_id is not None:
            data["login[integration_id]"] = login_integration_id

        self.logger.debug("PUT /api/v1/accounts/{account_id}/logins/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/accounts/{account_id}/logins/{id}".format(**path), data=data, params=params, no_data=True)