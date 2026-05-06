def bloquear_sat(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.bloquear_sat`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('bloquearsat')
        conteudo = resp.json()
        return RespostaSAT.bloquear_sat(conteudo.get('retorno'))