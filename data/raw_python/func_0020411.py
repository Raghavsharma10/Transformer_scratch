def get_devices(self):
        """
        Return a list of devices.
        Deprecated, use get_actors instead.
        """
        url = self.base_url + '/net/home_auto_query.lua'
        response = self.session.get(url, params={
            'sid': self.sid,
            'command': 'AllOutletStates',
            'xhr': 0,
        }, timeout=15)
        response.raise_for_status()
        data = response.json()
        count = int(data["Outlet_count"])
        devices = []
        for i in range(1, count + 1):
            device = Device(
                int(data["DeviceID_{0}".format(i)]),
                int(data["DeviceConnectState_{0}".format(i)]),
                int(data["DeviceSwitchState_{0}".format(i)])
            )
            devices.append(device)
        return devices