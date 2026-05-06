def get_uaa(self):
        """
        Returns an insstance of the UAA Service.
        """
        import predix.security.uaa
        uaa = predix.security.uaa.UserAccountAuthentication()
        return uaa