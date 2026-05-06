def consultar_numero_sessao(self, numero_sessao):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.consultar_numero_sessao`.

        :return: Uma resposta SAT que irá depender da sessão consultada.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('consultarnumerosessao',
                numero_sessao=numero_sessao)
        conteudo = resp.json()
        return RespostaConsultarNumeroSessao.analisar(conteudo.get('retorno'))