def getRealInterfaceNumber(self, interface):
        """
        Returns the host-visible interface number, or None if there is no such
        interface.
        """
        try:
            return self._ioctl(INTERFACE_REVMAP, interface)
        except IOError as exc:
            if exc.errno == errno.EDOM:
                return None
            raise