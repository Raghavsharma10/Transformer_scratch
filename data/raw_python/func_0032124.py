def fetch(self):
        """
        Fetch & return a new `Action` object representing the action's current
        state

        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return api._action(api.request(self.url)["action"])