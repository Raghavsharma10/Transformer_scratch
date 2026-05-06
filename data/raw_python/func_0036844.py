def validate_config(cls, config):
        """
        Runs a check on the given config to make sure that `port`/`ports` and
        `discovery` is defined.
        """
        if "discovery" not in config:
            raise ValueError("No discovery method defined.")

        if not any([item in config for item in ["port", "ports"]]):
            raise ValueError("No port(s) defined.")

        cls.validate_check_configs(config)