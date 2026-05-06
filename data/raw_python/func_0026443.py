def _login(self, event, user_account, user_profile, client_config):
        """Send login notification to client"""

        user_account.lastlogin = std_now()
        user_account.save()

        user_account.passhash = ""
        self.fireEvent(
            authentication(user_account.name, (
                user_account, user_profile, client_config),
                           event.clientuuid,
                           user_account.uuid,
                           event.sock),
            "auth")