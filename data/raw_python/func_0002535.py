def capture_on(pattern, callback):
        """
        :param pattern: a wildcard pattern to match the description of a network interface to capture packets on
        :param callback: a function to call with each intercepted packet
        """
        device_name, desc = WinPcapDevices.get_matching_device(pattern)
        if device_name is not None:
            with WinPcap(device_name) as capture:
                capture.run(callback=callback)