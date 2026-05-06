def submit(self, map, method, postfix):
        '''Realiza um requisição HTTP para a networkAPI.

        :param map: Dicionário com os dados para gerar o XML enviado no corpo da requisição HTTP.
        :param method: Método da requisição HTTP ('GET', 'POST', 'PUT' ou 'DELETE').
        :param postfix: Posfixo a ser colocado na URL básica de acesso à networkAPI. Ex: /ambiente

        :return: Tupla com o código e o corpo da resposta HTTP:
            (< codigo>, < descricao>)

        :raise NetworkAPIClientError: Erro durante a chamada HTTP para acesso à networkAPI.
        '''
        try:
            rest_request = RestRequest(
                self.get_url(postfix),
                method,
                self.user,
                self.password,
                self.user_ldap)
            return rest_request.submit(map)
        except RestError as e:
            raise ErrorHandler.handle(None, str(e))