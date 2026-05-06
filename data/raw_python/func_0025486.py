def trocar_codigo_de_ativacao(self, novo_codigo_ativacao,
            opcao=constantes.CODIGO_ATIVACAO_REGULAR,
            codigo_emergencia=None):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.trocar_codigo_de_ativacao`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        retorno = super(ClienteSATLocal, self).trocar_codigo_de_ativacao(
                novo_codigo_ativacao, opcao=opcao,
                codigo_emergencia=codigo_emergencia)
        return RespostaSAT.trocar_codigo_de_ativacao(retorno)