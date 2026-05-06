def login(self, **kwargs):
    """Logs the current user into the server with the passed in credentials. If successful the apiToken will be changed to match the passed in credentials.

    :param apiToken: use the passed apiToken to authenticate
    :param user_id: optional instead of apiToken, must be passed with token
    :param token: optional instead of apiToken, must be passed with user_id
    :param authenticate: only valid with apiToken. Force a call to the server to authenticate the passed credentials.
    :return:
    """
    if 'signed_username' in kwargs:
      apiToken = kwargs['signed_username']
      if kwargs.get('authenticate', False):
        self._checkReturn(requests.get("{}/users?signed_username={}".format(self.url, apiToken)))
      self.signedUsername = apiToken
    else:
      auth = (kwargs['user_id'], kwargs['token'])
      self.signedUsername = self._checkReturn(requests.get("{}/users/login".format(self.url), auth=auth))[
        'signed_username']