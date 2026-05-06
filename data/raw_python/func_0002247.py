def set_smartplug_state(self, device_label, state):
        """ Turn on or off smartplug

        Args:
            device_label (str): Smartplug device label
            state (boolean): new status, 'True' or 'False'
        """
        response = None
        try:
            response = requests.post(
                urls.smartplug(self._giid),
                headers={
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)},
                data=json.dumps([{
                    "deviceLabel": device_label,
                    "state": state}]))
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)