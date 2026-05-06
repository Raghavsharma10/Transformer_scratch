def get_appliance(self, appliance_id):
    """Get the information for a specified appliance

    Args:
      appliance_id (string): identifiying string of appliance

    Returns:
      list: dictionary object containing information about the specified appliance
    """
    url = "https://api.neur.io/v1/appliances/%s"%(appliance_id)

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    r = requests.get(url, headers=headers)
    return r.json()