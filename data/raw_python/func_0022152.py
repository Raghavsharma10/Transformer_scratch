def parse_auth_token(self, auth_token):
        """
        Break auth_token up into it's constituent values.

        **Parameters:**

          - **auth_token:** Auth_token string

        **Returns:** dict with Auth Token constituents
        """
        # remove the random security key value from the front of the auth_token
        auth_token_cleaned = auth_token.split('-', 1)[1]
        # URL Decode the Auth Token
        auth_token_decoded = self.url_decode(auth_token_cleaned)
        # Create a new dict to hold the response.
        auth_dict = {}

        # Parse the token
        for key_value in auth_token_decoded.split("&"):
            key_value_list = key_value.split("=")
            # check for valid token parts
            if len(key_value_list) == 2 and type(key_value_list[0]) in [text_type, binary_type]:
                auth_dict[key_value_list[0]] = key_value_list[1]

        # Return the dict of key/values in the token.
        return auth_dict