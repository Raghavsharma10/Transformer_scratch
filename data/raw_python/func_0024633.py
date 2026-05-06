def add(self, device):
        """Add device."""
        if not isinstance(device, Device):
            raise TypeError()
        self.__devices.append(device)