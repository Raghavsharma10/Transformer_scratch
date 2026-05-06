def bloquear_sat(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.bloquear_sat`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='BloquearSAT')
        if resposta.EEEEE not in ('16000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta