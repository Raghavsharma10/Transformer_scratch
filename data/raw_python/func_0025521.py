def ativar_sat(self, tipo_certificado, cnpj, codigo_uf):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.ativar_sat`.

        :return: Uma resposta SAT especializada em ``AtivarSAT``.
        :rtype: satcfe.resposta.ativarsat.RespostaAtivarSAT
        """
        resp = self._http_post('ativarsat',
                tipo_certificado=tipo_certificado,
                cnpj=cnpj,
                codigo_uf=codigo_uf)
        conteudo = resp.json()
        return RespostaAtivarSAT.analisar(conteudo.get('retorno'))