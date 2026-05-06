def atualizar_software_sat(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.atualizar_software_sat`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('atualizarsoftwaresat')
        conteudo = resp.json()
        return RespostaSAT.atualizar_software_sat(conteudo.get('retorno'))