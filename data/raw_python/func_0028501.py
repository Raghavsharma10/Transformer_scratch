def _get_token(self):
        """
            Get token for make request. The The data obtained herein are used 
            in the variable header.

            Returns:
                To perform the request, receive in return a dictionary
                with several keys. With this method only return the token
                as it will use it for subsequent requests, such as a 
                sentence translate. Returns one string type.
        """
        informations = self._set_format_oauth()
        oauth_url = "https://datamarket.accesscontrol.windows.net/v2/OAuth2-13"
        token = requests.post(oauth_url, informations).json()
        return token["access_token"]