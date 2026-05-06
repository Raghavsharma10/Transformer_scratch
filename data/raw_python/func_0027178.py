def get_authentication_header(self, user=None, api_key=None, password=None, certificate=None):
        """ Return authenication string to place in Authorization Header

            If API Token is set, it'll be used. Otherwise, the clear
            text password will be sent. Users of NURESTLoginController are responsible to
            clean the password property.

            Returns:
                Returns the XREST Authentication string with API Key or user password encoded.
        """

        if not user:
            user = self.user

        if not api_key:
            api_key = self.api_key

        if not password:
            password = self.password

        if not password:
            password = self.password

        if not certificate:
            certificate = self._certificate

        if certificate:
            return "XREST %s" % urlsafe_b64encode("{}:".format(user).encode('utf-8')).decode('utf-8')

        if api_key:
            return "XREST %s" % urlsafe_b64encode("{}:{}".format(user, api_key).encode('utf-8')).decode('utf-8')


        return "XREST %s" % urlsafe_b64encode("{}:{}".format(user, password).encode('utf-8')).decode('utf-8')