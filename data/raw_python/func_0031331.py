def get_appliances(self, location_id):
    """Get the appliances added for a specified location.

    Args:
      location_id (string): identifiying string of appliance

    Returns:
      list: dictionary objects containing appliances data
    """
    url = "https://api.neur.io/v1/appliances"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    params = {
      "locationId": location_id,
    }
    url = self.__append_url_params(url, params)

    r = requests.get(url, headers=headers)
    return r.json()