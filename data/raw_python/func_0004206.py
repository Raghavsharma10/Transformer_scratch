def inserir(
            self,
            id_grupo_usuario,
            id_grupo_equipamento,
            leitura,
            escrita,
            alterar_config,
            exclusao):
        """Cria um novo direito de um grupo de usuário em um grupo de equipamento e retorna o seu identificador.

        :param id_grupo_usuario: Identificador do grupo de usuário.
        :param id_grupo_equipamento: Identificador do grupo de equipamento.
        :param leitura: Indicação de permissão de leitura ('0' ou '1').
        :param escrita: Indicação de permissão de escrita ('0' ou '1').
        :param alterar_config: Indicação de permissão de alterar_config ('0' ou '1').
        :param exclusao: Indicação de permissão de exclusão ('0' ou '1').

        :return: Dicionário com a seguinte estrutura: {'direito_grupo_equipamento': {'id': < id>}}

        :raise InvalidParameterError: Pelo menos um dos parâmetros é nulo ou inválido.
        :raise GrupoEquipamentoNaoExisteError: Grupo de Equipamento não cadastrado.
        :raise GrupoUsuarioNaoExisteError: Grupo de Usuário não cadastrado.
        :raise ValorIndicacaoDireitoInvalidoError: Valor de leitura, escrita, alterar_config e/ou exclusão inválido.
        :raise DireitoGrupoEquipamentoDuplicadoError: Já existe direitos cadastrados para o grupo de usuário e grupo
            de equipamento informados.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        direito_map = dict()
        direito_map['id_grupo_usuario'] = id_grupo_usuario
        direito_map['id_grupo_equipamento'] = id_grupo_equipamento
        direito_map['leitura'] = leitura
        direito_map['escrita'] = escrita
        direito_map['alterar_config'] = alterar_config
        direito_map['exclusao'] = exclusao

        code, xml = self.submit(
            {'direito_grupo_equipamento': direito_map}, 'POST', 'direitosgrupoequipamento/')

        return self.response(code, xml)