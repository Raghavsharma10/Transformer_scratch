def alterar(self, id_egrupo, nome):
        """Altera os dados de um grupo de equipamento a partir do seu identificador.

        :param id_egrupo: Identificador do grupo de equipamento.
        :param nome: Nome do grupo de equipamento.

        :return: None

        :raise InvalidParameterError: O identificador e/ou o nome do grupo são nulos ou inválidos.
        :raise GrupoEquipamentoNaoExisteError: Grupo de equipamento não cadastrado.
        :raise NomeGrupoEquipamentoDuplicadoError: Nome do grupo de equipamento duplicado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        if not is_valid_int_param(id_egrupo):
            raise InvalidParameterError(
                u'O identificador do grupo de equipamento é inválido ou não foi informado.')

        url = 'egrupo/' + str(id_egrupo) + '/'

        egrupo_map = dict()
        egrupo_map['nome'] = nome

        code, xml = self.submit({'grupo': egrupo_map}, 'PUT', url)

        return self.response(code, xml)