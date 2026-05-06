def validate_config(cls, config):
        """
        Validates that required config entries are present.

        Each check requires a `host`, `port`, `rise` and `fall` to be
        configured.

        The rise and fall variables are integers denoting how many times a
        check must pass before being considered passing and how many times a
        check must fail before being considered failing.
        """
        if "rise" not in config:
            raise ValueError("No 'rise' configured")
        if "fall" not in config:
            raise ValueError("No 'fall' configured")

        cls.validate_check_config(config)