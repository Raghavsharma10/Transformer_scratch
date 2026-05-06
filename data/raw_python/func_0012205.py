def load_devices(self):
        """
        load stored devices from the local file
        """
        self._devices = []
        if os.path.exists(self._devices_filename):
            log.debug(
                "loading devices from '{}'...".format(self._devices_filename)
            )
            with codecs.open(self._devices_filename, "rb", "utf-8") as f:
                self._devices = json.load(f)

        return self._devices