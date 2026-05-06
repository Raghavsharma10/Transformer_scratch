def get_samples_live(self, sensor_id, last=None):
    """Get recent samples, one sample per second for up to the last 2 minutes.

    Args:
      sensor_id (string): hexadecimal id of the sensor to query, e.g.
        ``0x0013A20040B65FAD``
      last (string): starting range, as ISO8601 timestamp

    Returns:
      list: dictionary objects containing sample data
    """
    url = "https://api.neur.io/v1/samples/live"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    params = { "sensorId": sensor_id }
    if last:
      params["last"] = last
    url = self.__append_url_params(url, params)

    r = requests.get(url, headers=headers)
    return r.json()