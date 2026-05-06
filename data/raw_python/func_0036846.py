def apply_config(self, config):
        """
        Takes a given validated config dictionary and sets an instance
        attribute for each one.

        For check definitions, a Check instance is is created and a `checks`
        attribute set to a dictionary keyed off of the checks' names.  If
        the Check instance has some sort of error while being created an error
        is logged and the check skipped.
        """
        self.host = config.get("host", "127.0.0.1")

        self.configured_ports = config.get("ports", [config.get("port")])

        self.discovery = config["discovery"]

        self.metadata = config.get("metadata", {})

        self.update_ports()

        self.check_interval = config["checks"]["interval"]

        self.update_checks(config["checks"])