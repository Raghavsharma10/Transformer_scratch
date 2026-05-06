def configurar_interface_de_rede(self, configuracao):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.configurar_interface_de_rede`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('configurarinterfacederede',
                configuracao=configuracao.documento())
        conteudo = resp.json()
        return RespostaSAT.configurar_interface_de_rede(conteudo.get('retorno'))