def get_acs(self):
        """
        Returns an instance of the Asset Control Service.
        """
        import predix.security.acs
        acs = predix.security.acs.AccessControl()
        return acs