def _do_request(self, url, params=None, data=None, headers=None):
        """
        Realiza as requisições diversas utilizando a biblioteca requests,
        tratando de forma genérica as exceções.
        """

        if not headers:
            headers = {'content-type': 'application/json'}

        try:
            response = requests.get(
                url, params=params, data=data, headers=headers)
        except:
            return None

        if response.status_code == 200:
            return response