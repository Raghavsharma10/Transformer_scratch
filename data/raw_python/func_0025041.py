def _get_service_config(self):
        """
        Reads in config file of UAA credential information
        or generates one as a side-effect if not yet
        initialized.
        """
        # Should work for windows, osx, and linux environments
        if not os.path.exists(self.config_path):
            try:
                os.makedirs(os.path.dirname(self.config_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

            return {}

        with open(self.config_path, 'r') as data:
            return json.load(data)