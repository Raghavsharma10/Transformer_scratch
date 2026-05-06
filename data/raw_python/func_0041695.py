def get_access_token(self) -> str:
        """
        Returns the access token in case of successful authorization
        """
        if self._service_token:
            return self._service_token
        if self._app_id and self._login and self._password:
            try:
                if self.login():
                    url_params = self.auth_oauth2()
                    if 'access_token' in url_params:
                        return url_params['access_token']
            finally:
                self.close()