def enviar_dados_venda(self, dados_venda):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.enviar_dados_venda`.

        :return: Uma resposta SAT especializada em ``EnviarDadosVenda``.
        :rtype: satcfe.resposta.enviardadosvenda.RespostaEnviarDadosVenda
        """
        retorno = super(ClienteSATLocal, self).enviar_dados_venda(dados_venda)
        return RespostaEnviarDadosVenda.analisar(retorno)