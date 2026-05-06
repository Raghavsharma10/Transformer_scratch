def _evictStaleDevices(self):
        """
        A housekeeping function which runs in a worker thread and which evicts devices that haven't sent an update for a
        while.
        """
        while self.running:
            expiredDeviceIds = [key for key, value in self.devices.items() if value.hasExpired()]
            for key in expiredDeviceIds:
                logger.warning("Device timeout, removing " + key)
                del self.devices[key]
            time.sleep(1)
            # TODO send reset after a device fails
        logger.warning("DeviceCaretaker is now shutdown")