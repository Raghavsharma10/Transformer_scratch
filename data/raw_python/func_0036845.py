def validate_check_configs(cls, config):
        """
        Config validation specific to the health check options.

        Verifies that checks are defined along with an interval, and calls
        out to the `Check` class to make sure each individual check's config
        is valid.
        """
        if "checks" not in config:
            raise ValueError("No checks defined.")
        if "interval" not in config["checks"]:
            raise ValueError("No check interval defined.")

        for check_name, check_config in six.iteritems(config["checks"]):
            if check_name == "interval":
                continue

            Check.from_config(check_name, check_config)