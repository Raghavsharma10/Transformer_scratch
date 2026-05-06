def submit(self, map):
        '''Envia a requisição HTTP de acordo com os parâmetros informados no construtor.

        :param map: Dicionário com os dados do corpo da requisição.

        :return: Retorna uma tupla contendo:
            (< código de resposta http >, < corpo da resposta >).

        :raise ConnectionError: Falha na conexão com a networkAPI.
        :raise RestError: Falha no acesso à networkAPI.
        '''
        # print "Requição em %s %s com corpo: %s" % (self.method, self.url,
        # map)
        rest = Rest()
        if self.method == 'POST':
            code, response = rest.post_map(self.url, map, self.auth_map)
        elif self.method == 'PUT':
            code, response = rest.put_map(self.url, map, self.auth_map)
        elif self.method == 'GET':
            code, response = rest.get_map(self.url, self.auth_map)
        elif self.method == 'DELETE':
            code, response = rest.delete_map(self.url, map, self.auth_map)

        return code, response