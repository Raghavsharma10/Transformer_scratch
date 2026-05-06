def analisar(retorno):
        """Constrói uma :class:`RespostaCancelarUltimaVenda` a partir do
        retorno informado.

        :param unicode retorno: Retorno da função ``CancelarUltimaVenda``.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='EnviarDadosVenda',
                classe_resposta=RespostaCancelarUltimaVenda,
                campos=(
                        ('numeroSessao', int),
                        ('EEEEE', unicode),
                        ('CCCC', unicode),
                        ('mensagem', unicode),
                        ('cod', unicode),
                        ('mensagemSEFAZ', unicode),
                        ('arquivoCFeBase64', unicode),
                        ('timeStamp', as_datetime),
                        ('chaveConsulta', unicode),
                        ('valorTotalCFe', Decimal),
                        ('CPFCNPJValue', unicode),
                        ('assinaturaQRCODE', unicode),
                    ),
                campos_alternativos=[
                        # se a venda falhar apenas os primeiros seis campos
                        # especificados na ER deverão ser retornados...
                        (
                                ('numeroSessao', int),
                                ('EEEEE', unicode),
                                ('CCCC', unicode),
                                ('mensagem', unicode),
                                ('cod', unicode),
                                ('mensagemSEFAZ', unicode),
                        ),
                        # por via das dúvidas, considera o padrão de campos,
                        # caso não haja nenhuma coincidência...
                        RespostaSAT.CAMPOS,
                    ]
            )
        if resposta.EEEEE not in ('07000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta