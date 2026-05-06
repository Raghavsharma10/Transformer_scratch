def params(self):
        """Parameters used in the url of the API call and for authentication.

        :return: parameters used in the url.
        :rtype: dict
        """
        params = {}
        params["access_token"] = self.access_token
        params["account_id"] = self.account_id
        return params