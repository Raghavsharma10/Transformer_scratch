def reset(self):
        """ Reset controller

            It removes all information about previous session
        """

        self._is_impersonating = False
        self._impersonation = None

        self.user = None
        self.password = None
        self.api_key = None
        self.enterprise = None
        self.url = None