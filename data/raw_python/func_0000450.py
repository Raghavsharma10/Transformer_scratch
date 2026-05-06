def ready(self):
        """
        Checks if organization properly created.
        Note: New organization must have 'default' environment and two default services
        running there. Cannot use DEFAULT_ENV_NAME, because zone could be added there.
        :rtype: bool
        """

        @retry(tries=3, retry_exception=exceptions.NotFoundError)  # org init, takes some times
        def check_init():
            env = self.environments['default']
            return env.services['Default workflow service'].running(timeout=1) and \
                   env.services['Default credentials service'].running(timeout=1)
        return check_init()