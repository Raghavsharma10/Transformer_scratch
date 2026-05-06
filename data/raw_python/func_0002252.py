def get_lock_config(self, device_label):
        """ Get lock configuration

        Args:
            device_label (str): device label of lock
        """
        response = None
        try:
            response = requests.get(
                urls.lockconfig(self._giid, device_label),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Cookie': 'vid={}'.format(self._vid)})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)