def _load_json_config(self):
        """Load the configuration file in JSON format

        :rtype: dict

        """
        try:
            return json.loads(self._read_config())
        except ValueError as error:
            raise ValueError(
                'Could not read configuration file: {}'.format(error))