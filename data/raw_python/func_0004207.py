def alterar(self, id_direito, leitura, escrita, alterar_config, exclusao):
        """Altera os direitos de um grupo de usuário em um grupo de equipamento a partir do seu identificador.

        :param id_direito: Identificador do direito grupo equipamento.
        :param leitura: Indicação de permissão de leitura ('0' ou '1').
        :param escrita: Indicação de permissão de escrita ('0' ou '1').
        :param alterar_config: Indicação de permissão de alterar_config ('0' ou '1').
        :param exclusao: Indicação de permissão de exclusão ('0' ou '1').

        :return: None

        :raise InvalidParameterError: Pelo menos um dos parâmetros é nulo ou inválido.
        :raise ValorIndicacaoDireitoInvalidoError: Valor de leitura, escrita, alterar_config e/ou exclusão inválido.
        :raise DireitoGrupoEquipamentoNaoExisteError: Direito Grupo Equipamento não cadastrado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        if not is_valid_int_param(id_direito):
            raise InvalidParameterError(
                u'O identificador do direito grupo equipamento é inválido ou não foi informado.')

        url = 'direitosgrupoequipamento/' + str(id_direito) + '/'

        direito_map = dict()
        direito_map['leitura'] = leitura
        direito_map['escrita'] = escrita
        direito_map['alterar_config'] = alterar_config
        direito_map['exclusao'] = exclusao

        code, xml = self.submit(
            {'direito_grupo_equipamento': direito_map}, 'PUT', url)

        return self.response(code, xml)