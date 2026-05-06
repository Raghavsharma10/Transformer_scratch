async def refresh_token(self):
        """Refresh API token from KLF 200."""
        json_response = await self.api_call('auth', 'login', {'password': self.config.password}, add_authorization_token=False)
        if 'token' not in json_response:
            raise PyVLXException('no element token found in response: {0}'.format(json.dumps(json_response)))
        self.token = json_response['token']