def delete_map(self, url, map=None, auth_map=None):
        """Gera um XML a partir dos dados do dicionário e o envia através de uma requisição DELETE.

        :param url: URL para enviar a requisição HTTP.
        :param map: Dicionário com os dados do corpo da requisição HTTP.
        :param auth_map: Dicionário com as informações para autenticação na networkAPI.

        :return: Retorna uma tupla contendo:
            (< código de resposta http >, < corpo da resposta >).

        :raise ConnectionError: Falha na conexão com a networkAPI.
        :raise RestError: Falha no acesso à networkAPI.
        """
        xml = None
        if map is not None:
            xml = dumps_networkapi(map)
        response_code, content = self.delete(url, xml, 'text/plain', auth_map)
        return response_code, content