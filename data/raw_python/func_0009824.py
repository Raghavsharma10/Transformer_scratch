def register(self, device, callback):
        """Register a callback.

        device: device to be updated by subscription
        callback: callback for notification of changes
        """
        if not device:
            logger.error("Received an invalid device: %r", device)
            return

        logger.debug("Subscribing to events for %s", device.name)
        self._devices[device.vera_device_id].append(device)
        self._callbacks[device].append(callback)