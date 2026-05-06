def get_map(self, url, auth_map=None):
        """Envia uma requisição GET.

        :param url: URL para enviar a requisição HTTP.
        :param auth_map: Dicionário com as informações para autenticação na networkAPI.

        :return: Retorna uma tupla contendo:
            (< código de resposta http >, < corpo da resposta >).

        :raise ConnectionError: Falha na conexão com a networkAPI.
        :raise RestError: Falha no acesso à networkAPI.
        """
        response_code, content = self.get(url, auth_map)
        return response_code, content