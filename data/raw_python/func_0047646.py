def user_data(self, access_token, *args, **kwargs):
        """Loads user data from service"""

        return self.get_json(self.USER_INFO_URL, method="POST", headers=self._get_headers(access_token))