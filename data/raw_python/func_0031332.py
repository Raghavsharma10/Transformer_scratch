def get_appliance_event_after_time(self, location_id, since, per_page=None, page=None, min_power=None):
    """Get appliance events by location Id after defined time.

    Args:
      location_id (string): hexadecimal id of the sensor to query, e.g.
                          ``0x0013A20040B65FAD``
      since (string): ISO 8601 start time for getting the events that are created or updated after it.
        Maxiumim value allowed is 1 day from the current time.
      min_power (string): The minimum average power (in watts) for filtering.
        Only events with an average power above this value will be returned.
        (default: 400)
      per_page (string, optional): the number of returned results per page
        (min 1, max 500) (default: 10)
      page (string, optional): the page number to return (min 1, max 100000)
        (default: 1)

    Returns:
      list: dictionary objects containing appliance events meeting specified criteria
    """
    url = "https://api.neur.io/v1/appliances/events"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    params = {
      "locationId": location_id,
      "since": since
    }
    if min_power:
      params["minPower"] = min_power
    if per_page:
      params["perPage"] = per_page
    if page:
      params["page"] = page
    url = self.__append_url_params(url, params)

    r = requests.get(url, headers=headers)
    return r.json()