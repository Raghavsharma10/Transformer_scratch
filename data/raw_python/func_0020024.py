def scan(self):
        """Scan for devices on the bus and return a list of addresses."""
        devices = []
        diff = 65
        rom = False
        count = 0
        for _ in range(0xff):
            rom, diff = self._search_rom(rom, diff)
            if rom:
                count += 1
                if count > self.maximum_devices:
                    raise RuntimeError(
                        "Maximum device count of {} exceeded."\
                        .format(self.maximum_devices))
                devices.append(OneWireAddress(rom))
            if diff == 0:
                break
        return devices