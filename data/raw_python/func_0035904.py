def auth_proxy(self, method):
        """Authentication proxy for API requests.

        This is required because the API objects are naive of ``HelpScout``,
        so they would otherwise be unauthenticated.

        Args:
            method (callable): A method call that should be authenticated. It
             should accept a ``requests.Session`` as its first parameter,
             which should be used for the actual API call.

        Returns:
            mixed: The results of the authenticated callable.
        """
        def _proxy(*args, **kwargs):
            """The actual proxy, which instantiates and authenticates the API.

            Args:
                *args (mixed): Args to send to class instantiation.
                **kwargs (mixed): Kwargs to send to class instantiation.

            Returns:
                mixed: The result of the authenticated callable.
            """
            return method(self.session, *args, **kwargs)

        return _proxy