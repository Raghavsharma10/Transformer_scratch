def comunicar_certificado_icpbrasil(retorno):
        """Constrói uma :class:`RespostaSAT` para o retorno (unicode) da função
        :meth:`~satcfe.base.FuncoesSAT.comunicar_certificado_icpbrasil`.
        """
        resposta = analisar_retorno(forcar_unicode(retorno),
                funcao='ComunicarCertificadoICPBRASIL')
        if resposta.EEEEE not in ('05000',):
            raise ExcecaoRespostaSAT(resposta)
        return resposta