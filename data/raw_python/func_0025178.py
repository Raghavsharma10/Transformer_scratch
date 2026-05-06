def trocar_codigo_de_ativacao(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.trocar_codigo_de_ativacao`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='TrocarCodigoDeAtivacao')
        if resposta.EEEEE not in ('18000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta