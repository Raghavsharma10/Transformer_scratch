def desbloquear_sat(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.desbloquear_sat`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('desbloquearsat')
        conteudo = resp.json()
        return RespostaSAT.desbloquear_sat(conteudo.get('retorno'))