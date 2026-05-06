def set_lock_state(self, code, device_label, state):
        """ Lock or unlock

        Args:
            code (str): Lock code
            device_label (str): device label of lock
            state (str): 'lock' or 'unlock'
        """
        response = None
        try:
            response = requests.put(
                urls.set_lockstate(self._giid, device_label, state),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)},
                data=json.dumps({"code": str(code)}))
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)