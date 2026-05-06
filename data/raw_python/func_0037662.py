def create_user_login(self, user_id, account_id, login_unique_id, login_authentication_provider_id=None, login_integration_id=None, login_password=None, login_sis_user_id=None):
        """
        Create a user login.

        Create a new login for an existing user in the given account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - user[id]
        """The ID of the user to create the login for."""
        data["user[id]"] = user_id

        # REQUIRED - login[unique_id]
        """The unique ID for the new login."""
        data["login[unique_id]"] = login_unique_id

        # OPTIONAL - login[password]
        """The new login's password."""
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

        # OPTIONAL - login[authentication_provider_id]
        """The authentication provider this login is associated with. Logins
        associated with a specific provider can only be used with that provider.
        Legacy providers (LDAP, CAS, SAML) will search for logins associated with
        them, or unassociated logins. New providers will only search for logins
        explicitly associated with them. This can be the integer ID of the
        provider, or the type of the provider (in which case, it will find the
        first matching provider)."""
        if login_authentication_provider_id is not None:
            data["login[authentication_provider_id]"] = login_authentication_provider_id

        self.logger.debug("POST /api/v1/accounts/{account_id}/logins with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/logins".format(**path), data=data, params=params, no_data=True)