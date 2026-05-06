def _clear(cls):
        """This is only for testing pourposes, resets the DepotManager status

        This is to simplify writing test fixtures, resets the DepotManager global
        status and removes the informations related to the current configured depots
        and middleware.
        """
        cls._default_depot = None
        cls._depots = {}
        cls._middleware = None
        cls._aliases = {}