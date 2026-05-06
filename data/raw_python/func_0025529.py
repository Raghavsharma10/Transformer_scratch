def associar_assinatura(self, sequencia_cnpj, assinatura_ac):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.associar_assinatura`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('associarassinatura',
                sequencia_cnpj=sequencia_cnpj, assinatura_ac=assinatura_ac)
        # (!) resposta baseada na redação com efeitos até 31-12-2016
        conteudo = resp.json()
        return RespostaSAT.associar_assinatura(conteudo.get('retorno'))