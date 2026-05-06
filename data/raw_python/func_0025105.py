def enviar_dados_venda(self, dados_venda):
        """Função ``EnviarDadosVenda`` conforme ER SAT, item 6.1.3. Envia o
        CF-e de venda para o equipamento SAT, que o enviará para autorização
        pela SEFAZ.

        :param dados_venda: Uma instância de :class:`~satcfe.entidades.CFeVenda`
            ou uma string contendo o XML do CF-e de venda.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        cfe_venda = dados_venda \
                if isinstance(dados_venda, basestring) \
                else dados_venda.documento()

        return self.invocar__EnviarDadosVenda(
                self.gerar_numero_sessao(), self._codigo_ativacao, cfe_venda)