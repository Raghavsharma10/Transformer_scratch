def logout(self):
        """ Logout and remove vid """
        response = None
        try:
            response = requests.delete(
                urls.login(),
                headers={
                    'Cookie': 'vid={}'.format(self._vid)})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)