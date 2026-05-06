def handle(cls, code, description):
        '''Recebe o código e a descrição do erro da networkAPI e lança a exceção correspondente.

        :param code: Código de erro retornado pela networkAPI.
        :param description: Descrição do erro.

        :return: None
        '''
        if code is None:
            raise NetworkAPIClientError(description)

        if int(code) in cls.errors:
            raise cls.errors[int(code)](description)
        else:
            raise NetworkAPIClientError(description)