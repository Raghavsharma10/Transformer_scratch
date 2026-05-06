def set_heat_pump_mode(self, device_label, mode):
        """ Set heatpump mode
        Args:
            mode (str): 'HEAT', 'COOL', 'FAN' or 'AUTO'
        """
        response = None
        try:
            response = requests.put(
                urls.set_heatpump_state(self._giid, device_label),
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)},
                data=json.dumps({'mode': mode}))
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)