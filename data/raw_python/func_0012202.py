def get_devices(self, force_reload=False, save_devices=True):
        """
        get all devices that are linked to the user, if the local device
        file is not existing the devices will be obtained from the LaMetric
        cloud, otherwise the local device file will be read.

        :param bool force_reload: When True, devices are read again from cloud
        :param bool save_devices: When True, devices obtained from the LaMetric
                                  cloud are stored locally
        """
        if (
            (not os.path.exists(self._devices_filename)) or
            (force_reload is True)
        ):
            # -- load devices from LaMetric cloud --
            log.debug("getting devices from LaMetric cloud...")
            _, url = CLOUD_URLS["get_devices"]
            res = self._cloud_session.session.get(url)
            if res is not None:
                # raise an exception on error
                res.raise_for_status()

            # store obtained devices internally
            self._devices = res.json()
            if save_devices is True:
                # save obtained devices to the local file
                self.save_devices()

            return self._devices

        else:
            # -- load devices from local file --
            log.debug(
                "getting devices from '{}'...".format(self._devices_filename)
            )
            return self.load_devices()