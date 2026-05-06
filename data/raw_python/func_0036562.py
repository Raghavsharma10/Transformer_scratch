def apply_check_config(self, config):
        """
        Takes the `query` and `response` fields from a validated config
        dictionary and sets the proper instance attributes.
        """
        self.query = config.get("query")
        self.expected_response = config.get("response")