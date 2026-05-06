def analisar(retorno):
        """Constrói uma :class:`RespostaAtivarSAT` a partir do retorno
        informado.

        :param unicode retorno: Retorno da função ``AtivarSAT``.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='AtivarSAT',
                classe_resposta=RespostaAtivarSAT,
                campos=(
                        ('numeroSessao', int),
                        ('EEEEE', unicode),
                        ('mensagem', unicode),
                        ('cod', unicode),
                        ('mensagemSEFAZ', unicode),
                        ('CSR', unicode),
                    ),
                campos_alternativos=[
                        # se a ativação falhar espera-se o padrão de campos
                        # no retorno...
                        RespostaSAT.CAMPOS,
                    ]
            )
        if resposta.EEEEE not in (
                ATIVADO_CORRETAMENTE,
                CSR_ICPBRASIL_CRIADO_SUCESSO,):
            raise ExcecaoRespostaSAT(resposta)
        return resposta