def get_samples_live_last(self, sensor_id):
    """Get the last sample recorded by the sensor.

    Args:
      sensor_id (string): hexadecimal id of the sensor to query, e.g.
        ``0x0013A20040B65FAD``

    Returns:
      list: dictionary objects containing sample data
    """
    url = "https://api.neur.io/v1/samples/live/last"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    params = { "sensorId": sensor_id }
    url = self.__append_url_params(url, params)

    r = requests.get(url, headers=headers)
    return r.json()