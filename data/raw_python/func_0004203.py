def listar_por_grupo_usuario(self, id_grupo_usuario):
        """Lista todos os direitos de um grupo de usuário em grupos de equipamento.

        :param id_grupo_usuario: Identificador do grupo de usuário para filtrar a pesquisa.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'direito_grupo_equipamento':
            [{'id_grupo_equipamento': < id_grupo_equipamento >,
            'exclusao': < exclusao >,
            'alterar_config': < alterar_config >,
            'nome_grupo_equipamento': < nome_grupo_equipamento >,
            'id_grupo_usuario': < id_grupo_usuario >,
            'escrita': < escrita >,
            'nome_grupo_usuario': < nome_grupo_usuario >,
            'id': < id >,
            'leitura': < leitura >}, … demais direitos …]}

        :raise InvalidParameterError: O identificador do grupo de usuário é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_grupo_usuario):
            raise InvalidParameterError(
                u'O identificador do grupo de usuário é inválido ou não foi informado.')

        url = 'direitosgrupoequipamento/ugrupo/' + str(id_grupo_usuario) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'direito_grupo_equipamento'
        return get_list_map(self.response(code, map, [key]), key)