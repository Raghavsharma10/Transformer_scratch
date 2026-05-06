def extrair_logs(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.extrair_logs`.

        :return: Uma resposta SAT especializada em ``ExtrairLogs``.
        :rtype: satcfe.resposta.extrairlogs.RespostaExtrairLogs
        """
        resp = self._http_post('extrairlogs')
        conteudo = resp.json()
        return RespostaExtrairLogs.analisar(conteudo.get('retorno'))