def get_token(self):
    """Performs Neurio API token authentication using provided key and secret.

    Note:
      This method is generally not called by hand; rather it is usually
      called as-needed by a Neurio Client object.

    Returns:
      string: the access token
    """
    if self.__token is not None:
      return self.__token

    url = "https://api.neur.io/v1/oauth2/token"

    creds = b64encode(":".join([self.__key,self.__secret]).encode()).decode()

    headers = {
      "Authorization": " ".join(["Basic", creds]),
    }
    payload = {
      "grant_type": "client_credentials"
    }

    r = requests.post(url, data=payload, headers=headers)

    self.__token = r.json()["access_token"]

    return self.__token