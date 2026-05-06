def _login(self):
        """
        Login with your Google account
        :return:
        """
        # TODO(dmvieira) login changed to oauth2
        self.gc = self.gspread.login(self.email, self.password)