def apply_check_config(self, config):
        """
        Takes a validated config dictionary and sets the `uri`, `use_https`
        and `method` attributes based on the config's contents.
        """
        self.uri = config["uri"]
        self.use_https = config.get("https", False)
        self.method = config.get("method", "GET")