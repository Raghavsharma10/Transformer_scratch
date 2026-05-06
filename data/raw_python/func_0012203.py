def save_devices(self):
        """
        save devices that have been obtained from LaMetric cloud
        to a local file
        """
        log.debug("saving devices to ''...".format(self._devices_filename))
        if self._devices != []:
            with codecs.open(self._devices_filename, "wb", "utf-8") as f:
                json.dump(self._devices, f)