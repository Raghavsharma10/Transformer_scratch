def post(self, endpoint, message):
    """ Todo """
    r = self.http.request('POST',
                          self._api_base.format(endpoint),
                          headers={'Content-Type': 'application/json',
                                   'Authorization': 'Bot '+self.token},
                          body=message.encode('utf-8'))