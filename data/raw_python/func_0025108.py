def configurar_interface_de_rede(self, configuracao):
        """Função ``ConfigurarInterfaceDeRede`` conforme ER SAT, item 6.1.9.
        Configurção da interface de comunicação do equipamento SAT.

        :param configuracao: Instância de :class:`~satcfe.rede.ConfiguracaoRede`
            ou uma string contendo o XML com as configurações de rede.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        conf_xml = configuracao \
                if isinstance(configuracao, basestring) \
                else configuracao.documento()

        return self.invocar__ConfigurarInterfaceDeRede(
                self.gerar_numero_sessao(), self._codigo_ativacao, conf_xml)