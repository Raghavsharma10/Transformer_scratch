def enviar_dados_venda(self, dados_venda):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.enviar_dados_venda`.

        :return: Uma resposta SAT especializada em ``EnviarDadosVenda``.
        :rtype: satcfe.resposta.enviardadosvenda.RespostaEnviarDadosVenda
        """
        resp = self._http_post('enviardadosvenda',
                dados_venda=dados_venda.documento())
        conteudo = resp.json()
        return RespostaEnviarDadosVenda.analisar(conteudo.get('retorno'))