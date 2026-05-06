def _init(self):
        """Read the success byte."""
        self._api_version = self._file.read(1)[0]
        self._firmware_version = FirmwareVersion(*self._file.read(2))