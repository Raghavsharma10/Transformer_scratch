def _read_config(self):
        """Read the configuration from the various places it may be read from.

        :rtype: str
        :raises: ValueError

        """
        if not self._file_path:
            return None
        elif self._file_path.startswith('s3://'):
            return self._read_s3_config()
        elif self._file_path.startswith('http://') or \
                self._file_path.startswith('https://'):
            return self._read_remote_config()
        elif not path.exists(self._file_path):
            raise ValueError(
                'Configuration file not found: {}'.format(self._file_path))

        with open(self._file_path, 'r') as handle:
            return handle.read()