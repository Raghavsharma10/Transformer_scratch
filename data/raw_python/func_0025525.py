def consultar_sat(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.consultar_sat`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('consultarsat')
        conteudo = resp.json()
        return RespostaSAT.consultar_sat(conteudo.get('retorno'))