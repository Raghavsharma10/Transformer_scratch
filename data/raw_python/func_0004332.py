def remover_provisionamento(self, equipamentos, vips):
        """Remove o provisionamento de um grupo virtual para o sistema de Orquestração VM.

        :param equipamentos: Lista de equipamentos gerada pelo método "add_equipamento_remove" da
          classe "EspecificacaoGrupoVirtual".
        :param vips: Lista de VIPs gerada pelo método "add_vip_remove" da classe "EspecificacaoGrupoVirtual".

        :return: None

        :raise InvalidParameterError: Algum dado obrigatório não foi informado nas listas ou possui um valor inválido.
        :raise IpNaoExisteError: IP não cadastrado.
        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise IpError: IP não está associado ao equipamento.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        code, map = self.submit({'equipamentos': {'equipamento': equipamentos}, 'vips': {
                                'vip': vips}}, 'DELETE', 'grupovirtual/')

        return self.response(code, map)