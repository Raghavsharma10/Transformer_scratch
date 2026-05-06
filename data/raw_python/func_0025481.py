def associar_assinatura(self, sequencia_cnpj, assinatura_ac):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.associar_assinatura`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        retorno = super(ClienteSATLocal, self).\
                associar_assinatura(sequencia_cnpj, assinatura_ac)
        # (!) resposta baseada na redação com efeitos até 31-12-2016
        return RespostaSAT.associar_assinatura(retorno)