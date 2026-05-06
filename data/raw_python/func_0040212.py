def _commit(self):
        """
            Envia requisição de commit do indice através da API.
        """

        params = {'commit': 'true'}

        response = self._do_request(
            self.UPDATE_ENDPOINT,
            params=params
        )

        if response and response.status_code == 200:
            logger.debug('Index commited')
            return None

        logger.warning('Fail to commite index')