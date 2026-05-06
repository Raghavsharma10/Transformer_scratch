def get_bulb(self, mac):
        """
        Returns a Bulb object corresponding to the bulb with the mac address
        `mac` (a 6-byte bytestring).
        """
        return self.bulbs.get(mac, Bulb('Bulb %s' % _bytes(mac), mac))