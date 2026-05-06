def __set_web_authentication_detail(self):
        """
        Sets up the WebAuthenticationDetail node. This is required for all
        requests.
        """

        # Start of the authentication stuff.
        web_authentication_credential = self.client.factory.create('WebAuthenticationCredential')
        web_authentication_credential.Key = self.config_obj.key
        web_authentication_credential.Password = self.config_obj.password

        # Encapsulates the auth credentials.
        web_authentication_detail = self.client.factory.create('WebAuthenticationDetail')
        web_authentication_detail.UserCredential = web_authentication_credential

        # Set Default ParentCredential
        if hasattr(web_authentication_detail, 'ParentCredential'):
            web_authentication_detail.ParentCredential = web_authentication_credential

        self.WebAuthenticationDetail = web_authentication_detail