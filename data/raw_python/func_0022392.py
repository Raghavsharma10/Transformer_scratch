def logout(self, force=False):
        """
        Interactive logout - ensures uid/tid cleared so `cloudgenix.API` object/ requests.Session can be re-used.

        **Parameters:**:

          - **force**: Bool, force logout API call, even when using a static AUTH_TOKEN.

        **Returns:** Bool of whether the operation succeeded.
        """
        # Extract requests session for manipulation.
        session = self._parent_class.expose_session()

        # if force = True, or token_session = None/False, call logout API.
        if force or not self._parent_class.token_session:
            # Call Logout
            result = self._parent_class.get.logout()
            if result.cgx_status:
                # clear info from session.
                self._parent_class.tenant_id = None
                self._parent_class.tenant_name = None
                self._parent_class.is_esp = None
                self._parent_class.client_id = None
                self._parent_class.address_string = None
                self._parent_class.email = None
                self._parent_class._user_id = None
                self._parent_class._password = None
                self._parent_class.roles = None
                self._parent_class.token_session = None
                # Cookies are removed via LOGOUT API call. if X-Auth-Token set, clear.
                if session.headers.get('X-Auth-Token'):
                    self._parent_class.remove_header('X-Auth-Token')

            return result.cgx_status

        else:
            # Token Session and not forced.
            api_logger.debug('TOKEN SESSION, LOGOUT API NOT CALLED.')
            # clear info from session.
            self._parent_class.tenant_id = None
            self._parent_class.tenant_name = None
            self._parent_class.is_esp = None
            self._parent_class.client_id = None
            self._parent_class.address_string = None
            self._parent_class.email = None
            self._parent_class._user_id = None
            self._parent_class._password = None
            self._parent_class.roles = None
            self._parent_class.token_session = None
            # if X-Auth-Token set, clear.
            if session.headers.get('X-Auth-Token'):
                self._parent_class.remove_header('X-Auth-Token')

            return True