def response(self, code, xml, force_list=None):
        """Cria um dicionário com os dados de retorno da requisição HTTP ou
            lança uma exceção correspondente ao erro ocorrido.

        Se a requisição HTTP retornar o código 200 então este método retorna o
            dicionário com os dados da resposta.
        Se a requisição HTTP retornar um código diferente de 200 então este método
            lança uma exceção correspondente ao erro.

        Todas as exceções lançadas por este método deverão herdar de NetworkAPIClientError.

        :param code: Código de retorno da requisição HTTP.
        :param xml: XML ou descrição (corpo) da resposta HTTP.
        :param force_list: Lista com as tags do XML de resposta que deverão ser transformadas
            obrigatoriamente em uma lista no dicionário de resposta.

        :return: Dicionário com os dados da resposta HTTP retornada pela networkAPI.
        """
        if int(code) == 200:
            # Retorna o map
            return loads(xml, force_list)['networkapi']
        elif int(code) == 500:
            code, description = self.get_error(xml)
            return ErrorHandler.handle(code, description)
        else:
            return ErrorHandler.handle(code, xml)