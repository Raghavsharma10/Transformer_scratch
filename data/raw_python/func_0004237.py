def put(self, url, request_data, content_type=None, auth_map=None):
        """Envia uma requisição PUT para a URL informada.

        Se auth_map é diferente de None, então deverá conter as
            chaves NETWORKAPI_PASSWORD e NETWORKAPI_USERNAME para realizar
            a autenticação na networkAPI.

        As chaves e os seus valores são enviados no header da requisição.

        :param url: URL para enviar a requisição HTTP.
        :param request_data: Descrição para enviar no corpo da requisição HTTP.
        :param content_type: Tipo do conteúdo enviado em request_data. O valor deste
            parâmetro será adicionado no header "Content-Type" da requisição.
        :param auth_map: Dicionário com as informações para autenticação na networkAPI.

        :return: Retorna uma tupla contendo:

        ::

            (< código de resposta http >, < corpo da resposta >).

        :raise ConnectionError: Falha na conexão com a networkAPI.
        :raise RestError: Falha no acesso à networkAPI.
        """
        try:
            LOG.debug('PUT %s\n%s', url, request_data)
            parsed_url = urlparse(url)
            if parsed_url.scheme == 'https':
                connection = HTTPSConnection(
                    parsed_url.hostname,
                    parsed_url.port)
            else:
                connection = HTTPConnection(
                    parsed_url.hostname,
                    parsed_url.port)

            try:
                headers_map = dict()
                if auth_map is not None:
                    headers_map.update(auth_map)

                if content_type is not None:
                    headers_map['Content-Type'] = content_type

                connection.request(
                    'PUT',
                    parsed_url.path,
                    request_data,
                    headers_map)

                response = connection.getresponse()
                body = response.read()
                LOG.debug('PUT %s returns %s\n%s', url, response.status, body)
                return response.status, body
            finally:
                connection.close()
        except URLError as e:
            raise ConnectionError(e)
        except Exception as e:
            raise RestError(e, e.message)