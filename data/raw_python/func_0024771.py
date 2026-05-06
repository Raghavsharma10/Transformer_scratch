async def api_call(self, verb, action, params=None, add_authorization_token=True, retry=False):
        """Send api call."""
        if add_authorization_token and not self.token:
            await self.refresh_token()

        try:
            return await self._api_call_impl(verb, action, params, add_authorization_token)
        except InvalidToken:
            if not retry and add_authorization_token:
                await self.refresh_token()
                # Recursive call of api_call
                return await self.api_call(verb, action, params, add_authorization_token, True)
            raise