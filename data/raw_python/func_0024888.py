def _get_or_create_uaa(self, uaa):
        """
        Returns a valid UAA instance for performing administrative functions
        on services.
        """
        if isinstance(uaa, predix.admin.uaa.UserAccountAuthentication):
            return uaa

        logging.debug("Initializing a new UAA")
        return predix.admin.uaa.UserAccountAuthentication()