def _write_service_config(self):
        """
        Will write the config out to disk.
        """
        with open(self.config_path, 'w') as output:
            output.write(json.dumps(self.data, sort_keys=True, indent=4))