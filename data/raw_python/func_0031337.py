def get_samples(self, sensor_id, start, granularity, end=None,
                  frequency=None, per_page=None, page=None,
                  full=False):
    """Get a sensor's samples for a specified time interval.

    Args:
      sensor_id (string): hexadecimal id of the sensor to query, e.g.
                          ``0x0013A20040B65FAD``
      start (string): ISO 8601 start time of sampling; depends on the
        ``granularity`` parameter value, the maximum supported time ranges are:
        1 day for minutes or hours granularities, 1 month for days,
        6 months for weeks, 1 year for months granularity, and 10 years for
        years granularity
      granularity (string): granularity of the sampled data; must be one of
        "minutes", "hours", "days", "weeks", "months", or "years"
      end (string, optional): ISO 8601 stop time for sampling; should be later
        than start time (default: the current time)
      frequency (string, optional): frequency of the sampled data (e.g. with
        granularity set to days, a value of 3 will result in a sample for every
        third day, should be a multiple of 5 when using minutes granularity)
        (default: 1) (example: "1, 5")
      per_page (string, optional): the number of returned results per page
        (min 1, max 500) (default: 10)
      page (string, optional): the page number to return (min 1, max 100000)
        (default: 1)
      full (bool, optional): include additional information per sample
        (default: False)

    Returns:
      list: dictionary objects containing sample data
    """
    url = "https://api.neur.io/v1/samples"
    if full:
      url = "https://api.neur.io/v1/samples/full"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    params = {
      "sensorId": sensor_id,
      "start": start,
      "granularity": granularity
    }
    if end:
      params["end"] = end
    if frequency:
      params["frequency"] = frequency
    if per_page:
      params["perPage"] = per_page
    if page:
      params["page"] = page
    url = self.__append_url_params(url, params)

    r = requests.get(url, headers=headers)
    return r.json()