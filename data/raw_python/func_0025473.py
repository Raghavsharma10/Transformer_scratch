def ativar_sat(self, tipo_certificado, cnpj, codigo_uf):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.ativar_sat`.

        :return: Uma resposta SAT especilizada em ``AtivarSAT``.
        :rtype: satcfe.resposta.ativarsat.RespostaAtivarSAT
        """
        retorno = super(ClienteSATLocal, self).ativar_sat(
                tipo_certificado, cnpj, codigo_uf)
        return RespostaAtivarSAT.analisar(retorno)