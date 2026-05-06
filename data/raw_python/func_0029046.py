def _read_remote_config(self):
        """Read a remote config via URL.

        :rtype: str
        :raises: ValueError

        """
        try:
            import requests
        except ImportError:
            requests = None
        if not requests:
            raise ValueError(
                'Remote config URL specified but requests not installed')
        result = requests.get(self._file_path)
        if not result.ok:
            raise ValueError(
                'Failed to retrieve remote config: {}'.format(
                    result.status_code))
        return result.text