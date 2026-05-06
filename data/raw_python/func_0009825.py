def unregister(self, device, callback):
        """Remove a registered a callback.

        device: device that has the subscription
        callback: callback used in original registration
        """
        if not device:
            logger.error("Received an invalid device: %r", device)
            return

        logger.debug("Removing subscription for {}".format(device.name))
        self._callbacks[device].remove(callback)
        self._devices[device.vera_device_id].remove(device)