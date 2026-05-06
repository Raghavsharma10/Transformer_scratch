def validate(self):
        """
        Check the value of the config attributes.
        """
        for client in self.clients:
            for key in REQUIRED_KEYS:
                if key not in client:
                    raise MissingConfigValue(key)

            if 'revision_file' not in client:
                client.revision_file = DEFAULT_REVISION_FILEPATH.format(
                    client.key
                )