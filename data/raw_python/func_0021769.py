def authenticate(self, dn='', password=''):
        """
        Attempt to authenticate given dn and password using a bind operation.
        Return True if the bind is successful, and return False there was an
        exception raised that is contained in
        self.failed_authentication_exceptions.
        """
        try:
            self.connection.simple_bind_s(dn, password)
        except tuple(self.failed_authentication_exceptions):
            return False
        else:
            return True