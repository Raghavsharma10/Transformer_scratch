def set_heat_pump_feature(self, device_label, feature):
        """ Set heatpump mode
        Args:
            feature: 'QUIET', 'ECONAVI', or 'POWERFUL'
        """
        response = None
        try:
            response = requests.put(
                urls.set_heatpump_feature(self._giid, device_label, feature),
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)