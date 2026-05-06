def get_climate(self, device_label):
        """ Get climate history
        Args:
            device_label: device label of climate device
        """
        response = None
        try:
            response = requests.get(
                urls.climate(self._giid),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Cookie': 'vid={}'.format(self._vid)},
                params={
                    "deviceLabel": device_label})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)