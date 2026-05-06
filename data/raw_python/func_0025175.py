def atualizar_software_sat(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.atualizar_software_sat`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='AtualizarSoftwareSAT')
        if resposta.EEEEE not in ('14000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta