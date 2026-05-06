def consultar_sat(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.consultar_sat`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='ConsultarSAT')
        if resposta.EEEEE not in ('08000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta