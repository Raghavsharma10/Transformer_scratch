def associar_assinatura(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.associar_assinatura`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='AssociarAssinatura')
        if resposta.EEEEE not in ('13000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta