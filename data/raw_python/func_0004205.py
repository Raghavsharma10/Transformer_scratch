def buscar_por_id(self, id_direito):
        """Obtém os direitos de um grupo de usuário e um grupo de equipamento.

        :param id_direito: Identificador do direito grupo equipamento.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'direito_grupo_equipamento':
            {'id_grupo_equipamento': < id_grupo_equipamento >,
            'exclusao': < exclusao >,
            'alterar_config': < alterar_config >,
            'nome_grupo_equipamento': < nome_grupo_equipamento >,
            'id_grupo_usuario': < id_grupo_usuario >,
            'escrita': < escrita >,
            'nome_grupo_usuario': < nome_grupo_usuario >,
            'id': < id >,
            'leitura': < leitura >}}

        :raise InvalidParameterError: O identificador do direito grupo equipamento é nulo ou inválido.
        :raise DireitoGrupoEquipamentoNaoExisteError: Direito Grupo Equipamento não cadastrado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_direito):
            raise InvalidParameterError(
                u'O identificador do direito grupo equipamento é inválido ou não foi informado.')

        url = 'direitosgrupoequipamento/' + str(id_direito) + '/'

        code, map = self.submit(None, 'GET', url)

        return self.response(code, map)