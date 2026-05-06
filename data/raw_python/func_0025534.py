def trocar_codigo_de_ativacao(self, novo_codigo_ativacao,
            opcao=constantes.CODIGO_ATIVACAO_REGULAR,
            codigo_emergencia=None):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.trocar_codigo_de_ativacao`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('trocarcodigodeativacao',
                novo_codigo_ativacao=novo_codigo_ativacao,
                opcao=opcao,
                codigo_emergencia=codigo_emergencia)
        conteudo = resp.json()
        return RespostaSAT.trocar_codigo_de_ativacao(conteudo.get('retorno'))