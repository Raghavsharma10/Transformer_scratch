def set_auth_service(self, auth_service: BaseAuthService):
        """
        Sets the authentication service
        :param auth_service: BaseAuthService Authentication service
        :raises: TypeError If the auth_service object is not a subclass of rinzler.auth.BaseAuthService
        :rtype: Rinzler
        """
        if issubclass(auth_service.__class__, BaseAuthService):
            self.auth_service = auth_service
            return self
        else:
            raise TypeError("Your auth service object must be a subclass of rinzler.auth.BaseAuthService.")