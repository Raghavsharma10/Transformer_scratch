def get(self, endpoint):
    """ Todo """
    r = self.http.request('GET',
                          self._api_base.format(endpoint),
                          headers={'Authorization': 'Bot '+self.token})
    if r.status == 200:
      return json.loads(r.data.decode('utf-8'))
    else:
      return {}