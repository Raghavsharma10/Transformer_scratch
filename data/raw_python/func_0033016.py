def rootChild_resetPassword(self, req, webViewer):
        """
        Return a page which will allow the user to re-set their password.
        """
        from xmantissa.signup import PasswordResetResource
        return PasswordResetResource(self.store)