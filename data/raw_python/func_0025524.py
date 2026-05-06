def cancelar_ultima_venda(self, chave_cfe, dados_cancelamento):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.cancelar_ultima_venda`.

        :return: Uma resposta SAT especializada em ``CancelarUltimaVenda``.
        :rtype: satcfe.resposta.cancelarultimavenda.RespostaCancelarUltimaVenda
        """
        resp = self._http_post('cancelarultimavenda',
                chave_cfe=chave_cfe,
                dados_cancelamento=dados_cancelamento.documento())
        conteudo = resp.json()
        return RespostaCancelarUltimaVenda.analisar(conteudo.get('retorno'))