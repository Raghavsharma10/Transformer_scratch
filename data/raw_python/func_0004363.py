def listar_por_tipo_ambiente(self, id_tipo_equipamento, id_ambiente):
        """Lista os equipamentos de um tipo e que estão associados a um ambiente.

        :param id_tipo_equipamento: Identificador do tipo do equipamento.
        :param id_ambiente: Identificador do ambiente.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'equipamento': [{'id': < id_equipamento >,
            'nome': < nome_equipamento >,
            'id_tipo_equipamento': < id_tipo_equipamento >,
            'nome_tipo_equipamento': < nome_tipo_equipamento >,
            'id_modelo': < id_modelo >,
            'nome_modelo': < nome_modelo >,
            'id_marca': < id_marca >,
            'nome_marca': < nome_marca >
            }, ... demais equipamentos ...]}

        :raise InvalidParameterError: O identificador do tipo de equipamento e/ou do ambiente são nulos ou inválidos.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if not is_valid_int_param(id_tipo_equipamento):
            raise InvalidParameterError(
                u'O identificador do tipo do equipamento é inválido ou não foi informado.')

        if not is_valid_int_param(id_ambiente):
            raise InvalidParameterError(
                u'O identificador do ambiente é inválido ou não foi informado.')

        url = 'equipamento/tipoequipamento/' + \
            str(id_tipo_equipamento) + '/ambiente/' + str(id_ambiente) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'equipamento'
        return get_list_map(self.response(code, xml, [key]), key)