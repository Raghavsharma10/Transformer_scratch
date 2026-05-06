def capture_on_device_name(device_name, callback):
        """
        :param device_name: the name (guid) of a device as provided by WinPcapDevices.list_devices()
        :param callback: a function to call with each intercepted packet
        """
        with WinPcap(device_name) as capture:
            capture.run(callback=callback)