def set_device(self, dev):
        """
        set the current device (that will be used for following API calls)

        :param dict dev: device that should be used for the API calls
                         (can be obtained via get_devices function)
        """
        log.debug("setting device to '{}'".format(dev))
        self.dev = dev
        self.set_apps_list()