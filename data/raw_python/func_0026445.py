def _handle_autologin(self, event):
        """Automatic logins for client configurations that allow it"""

        self.log("Verifying automatic login request")

        # TODO: Check for a common secret

        # noinspection PyBroadException
        try:
            client_config = objectmodels['client'].find_one({
                'uuid': event.requestedclientuuid
            })
        except Exception:
            client_config = None

        if client_config is None or client_config.autologin is False:
            self.log("Autologin failed:", event.requestedclientuuid,
                     lvl=error)
            self._fail(event)
            return

        try:
            user_account = objectmodels['user'].find_one({
                'uuid': client_config.owner
            })
            if user_account is None:
                raise AuthenticationError
            self.log("Autologin for", user_account.name, lvl=debug)
        except Exception as e:
            self.log("No user object due to error: ", e, type(e),
                     lvl=error)
            self._fail(event)
            return

        if user_account.active is False:
            self.log("Account deactivated.")
            self._fail(event, 'Account deactivated.')
            return

        user_profile = self._get_profile(user_account)

        self._login(event, user_account, user_profile, client_config)

        self.log("Autologin successful!", lvl=warn)