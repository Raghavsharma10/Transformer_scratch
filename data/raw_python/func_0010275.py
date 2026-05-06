def get_mac_last4(self, use_cached=True):
        """Get the last 4 characters in the device mac address hex (e.g. 00:40:9D:58:17:5B -> 175B)

        This is useful for use as a short reference to the device.  It is not guaranteed to
        be unique (obviously) but will often be if you don't have too many devices.

        """
        chunks = self.get_mac(use_cached).split(":")
        mac4 = "%s%s" % (chunks[-2], chunks[-1])
        return mac4.upper()