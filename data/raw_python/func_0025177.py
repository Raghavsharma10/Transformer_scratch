def desbloquear_sat(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.desbloquear_sat`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='DesbloquearSAT')
        if resposta.EEEEE not in ('17000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta