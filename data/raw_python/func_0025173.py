def configurar_interface_de_rede(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.configurar_interface_de_rede`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='ConfigurarInterfaceDeRede')
        if resposta.EEEEE not in ('12000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta