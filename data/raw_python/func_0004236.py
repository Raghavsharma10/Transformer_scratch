def get(self, url, auth_map=None):
        """Envia uma requisição GET para a URL informada.

        Se auth_map é diferente de None, então deverá conter as
        chaves NETWORKAPI_PASSWORD e NETWORKAPI_USERNAME para realizar
        a autenticação na networkAPI.
        As chaves e os seus valores são enviados no header da requisição.

        :param url: URL para enviar a requisição HTTP.
        :param auth_map: Dicionário com as informações para autenticação na networkAPI.

        :return: Retorna uma tupla contendo:
            (< código de resposta http >, < corpo da resposta >).

        :raise ConnectionError: Falha na conexão com a networkAPI.
        :raise RestError: Falha no acesso à networkAPI.
        """
        try:
            LOG.debug('GET %s', url)
            request = Request(url)
            if auth_map is not None:
                for key in auth_map.iterkeys():
                    request.add_header(key, auth_map[key])
                # request.add_header('NETWORKAPI_PASSWORD', auth_map['NETWORKAPI_PASSWORD'])
                # request.add_header('NETWORKAPI_USERNAME', auth_map['NETWORKAPI_USERNAME'])
            content = urlopen(request).read()
            response_code = 200
            LOG.debug('GET %s returns %s\n%s', url, response_code, content)
            return response_code, content
        except HTTPError as e:
            response_code = e.code
            content = ''
            if int(e.code) == 500:
                content = e.read()
            LOG.debug('GET %s returns %s\n%s', url, response_code, content)
            return response_code, content
        except URLError as e:
            raise ConnectionError(e)
        except Exception as e:
            raise RestError(e, e.message)