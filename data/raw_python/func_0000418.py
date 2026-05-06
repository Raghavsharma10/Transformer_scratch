def read_sensor(self, device_id, sensor_uri):
        """Return sensor value based on sensor_uri."""
        url = MINUT_DEVICES_URL + "/{device_id}/{sensor_uri}".format(
            device_id=device_id, sensor_uri=sensor_uri)
        res = self._request(url, request_type='GET', data={'limit': 1})
        if not res.get('values'):
            return None
        return res.get('values')[-1].get('value')