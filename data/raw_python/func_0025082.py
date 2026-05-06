def analisar(retorno):
        """Constrói uma :class:`RespostaExtrairLogs` a partir do retorno
        informado.

        :param unicode retorno: Retorno da função ``ExtrairLogs``.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='ExtrairLogs',
                classe_resposta=RespostaExtrairLogs,
                campos=RespostaSAT.CAMPOS + (
                        ('arquivoLog', unicode),
                    ),
                campos_alternativos=[
                        # se a extração dos logs falhar espera-se o padrão de
                        # campos no retorno...
                        RespostaSAT.CAMPOS,
                    ]
            )
        if resposta.EEEEE not in ('15000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta