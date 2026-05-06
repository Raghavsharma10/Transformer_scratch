def listar_por_nome(self, nome):
        """Obtém um equipamento a partir do seu nome.

        :param nome: Nome do equipamento.

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
            nome informado não cadastrado.
        :raise InvalidParameterError: O nome do equipamento é nulo ou vazio.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if nome == '' or nome is None:
            raise InvalidParameterError(
                u'O nome do equipamento não foi informado.')

        url = 'equipamento/nome/' + urllib.quote(nome) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)