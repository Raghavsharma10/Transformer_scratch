def get_local_current_sample(ip):
    """Gets current sample from *local* Neurio device IP address.

    This is a static method. It doesn't require a token to authenticate.

    Note, call get_user_information to determine local Neurio IP addresses.

    Args:
      ip (string): address of local Neurio device

    Returns:
      dictionary object containing current sample information
    """
    valid_ip_pat = re.compile(
      "^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )
    if not valid_ip_pat.match(ip):
      raise ValueError("ip address invalid")

    url = "http://%s/current-sample" % (ip)
    headers = { "Content-Type": "application/json" }

    r = requests.get(url, headers=headers)
    return r.json()