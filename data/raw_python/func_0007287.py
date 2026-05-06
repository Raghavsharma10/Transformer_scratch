def whoami(self):
        """
        Get information about the access token.

        Official docs:
            https://monzo.com/docs/#authenticating-requests

        :returns: access token details
        :rtype: dict
        """
        endpoint = '/ping/whoami'
        response = self._get_response(
            method='get', endpoint=endpoint,
        )

        return response.json()