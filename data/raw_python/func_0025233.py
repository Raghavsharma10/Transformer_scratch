def analisar(retorno):
        """Constrói uma :class:`RespostaConsultarStatusOperacional` a partir do
        retorno informado.

        :param unicode retorno: Retorno da função ``ConsultarStatusOperacional``.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='ConsultarStatusOperacional',
                classe_resposta=RespostaConsultarStatusOperacional,
                campos=RespostaSAT.CAMPOS + (
                        ('NSERIE', as_clean_unicode),
                        ('TIPO_LAN', as_clean_unicode),
                        ('LAN_IP', normalizar_ip),
                        ('LAN_MAC', unicode),
                        ('LAN_MASK', normalizar_ip),
                        ('LAN_GW', normalizar_ip),
                        ('LAN_DNS_1', normalizar_ip),
                        ('LAN_DNS_2', normalizar_ip),
                        ('STATUS_LAN', as_clean_unicode),
                        ('NIVEL_BATERIA', as_clean_unicode),
                        ('MT_TOTAL', as_clean_unicode),
                        ('MT_USADA', as_clean_unicode),
                        ('DH_ATUAL', as_datetime),
                        ('VER_SB', as_clean_unicode),
                        ('VER_LAYOUT', as_clean_unicode),
                        ('ULTIMO_CF_E_SAT', as_clean_unicode),
                        ('LISTA_INICIAL', as_clean_unicode),
                        ('LISTA_FINAL', as_clean_unicode),
                        ('DH_CFE', as_datetime_or_none),
                        ('DH_ULTIMA', as_datetime),
                        ('CERT_EMISSAO', as_date),
                        ('CERT_VENCIMENTO', as_date),
                        ('ESTADO_OPERACAO', int),
                    ),
                campos_alternativos=[
                        # se falhar resultarão apenas os 5 campos padrão
                        RespostaSAT.CAMPOS,
                    ]
            )
        if resposta.EEEEE not in ('10000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta