def get_actors(self):
        """
        Returns a list of Actor objects for querying SmartHome devices.

        This is currently the only working method for getting temperature data.
        """
        devices = self.homeautoswitch("getdevicelistinfos")
        xml = ET.fromstring(devices)

        actors = []
        for device in xml.findall('device'):
            actors.append(Actor(fritzbox=self, device=device))

        return actors