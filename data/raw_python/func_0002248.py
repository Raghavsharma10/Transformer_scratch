def get_history(self, filters=(), pagesize=15, offset=0):
        """ Get recent events

        Args:
            filters (string set): 'ARM', 'DISARM', 'FIRE', 'INTRUSION',
                                  'TECHNICAL', 'SOS', 'WARNING', 'LOCK',
                                  'UNLOCK'
            pagesize (int): Number of events to display
            offset (int): Skip pagesize * offset first events
        """
        response = None
        try:
            response = requests.get(
                urls.history(self._giid),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Cookie': 'vid={}'.format(self._vid)},
                params={
                    "offset": int(offset),
                    "pagesize": int(pagesize),
                    "notificationCategories": filters})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)