def get_user_information(self):
    """Gets the current user information, including sensor ID

    Args:
      None

    Returns:
      dictionary object containing information about the current user
    """
    url = "https://api.neur.io/v1/users/current"

    headers = self.__gen_headers()
    headers["Content-Type"] = "application/json"

    r = requests.get(url, headers=headers)
    return r.json()