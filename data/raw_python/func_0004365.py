def listar_por_id(self, id):
        """Obtém um equipamento a partir do seu identificador.

        :param id: ID do equipamento.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'equipamento': {'id': < id_equipamento >,
            'nome': < nome_equipamento >,
            'id_tipo_equipamento': < id_tipo_equipamento >,
            'nome_tipo_equipamento': < nome_tipo_equipamento >,
            'id_modelo': < id_modelo >,
            'nome_modelo': < nome_modelo >,
            'id_marca': < id_marca >,
            'nome_marca': < nome_marca >}}

        :raise EquipamentoNaoExisteError: Equipamento com o
            id informado não cadastrado.
        :raise InvalidParameterError: O nome do equipamento é nulo ou vazio.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if id is None:
            raise InvalidParameterError(
                u'O id do equipamento não foi informado.')

        url = 'equipamento/id/' + urllib.quote(id) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)