def comunicar_certificado_icpbrasil(self, certificado):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.comunicar_certificado_icpbrasil`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        resp = self._http_post('comunicarcertificadoicpbrasil',
                certificado=certificado)
        conteudo = resp.json()
        return RespostaSAT.comunicar_certificado_icpbrasil(
                conteudo.get('retorno'))