def cancelar_ultima_venda(self, chave_cfe, dados_cancelamento):
        """Função ``CancelarUltimaVenda`` conforme ER SAT, item 6.1.4. Envia o
        CF-e de cancelamento para o equipamento SAT, que o enviará para
        autorização e cancelamento do CF-e pela SEFAZ.

        :param chave_cfe: String contendo a chave do CF-e a ser cancelado,
            prefixada com o literal ``CFe``.

        :param dados_cancelamento: Uma instância
            de :class:`~satcfe.entidades.CFeCancelamento` ou uma string
            contendo o XML do CF-e de cancelamento.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        cfe_canc = dados_cancelamento \
                if isinstance(dados_cancelamento, basestring) \
                else dados_cancelamento.documento()

        return self.invocar__CancelarUltimaVenda(
                self.gerar_numero_sessao(), self._codigo_ativacao,
                chave_cfe, cfe_canc)