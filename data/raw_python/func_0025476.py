def cancelar_ultima_venda(self, chave_cfe, dados_cancelamento):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.cancelar_ultima_venda`.

        :return: Uma resposta SAT especializada em ``CancelarUltimaVenda``.
        :rtype: satcfe.resposta.cancelarultimavenda.RespostaCancelarUltimaVenda
        """
        retorno = super(ClienteSATLocal, self).\
                cancelar_ultima_venda(chave_cfe, dados_cancelamento)
        return RespostaCancelarUltimaVenda.analisar(retorno)