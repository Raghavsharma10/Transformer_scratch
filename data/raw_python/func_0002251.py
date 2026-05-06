def get_lock_state_transaction(self, transaction_id):
        """ Get lock state transaction status

        Args:
            transaction_id: Transaction ID received from set_lock_state
        """
        response = None
        try:
            response = requests.get(
                urls.get_lockstate_transaction(self._giid, transaction_id),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Cookie': 'vid={}'.format(self._vid)})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)