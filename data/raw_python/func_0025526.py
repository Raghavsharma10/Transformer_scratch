def consultar_status_operacional(self):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.consultar_status_operacional`.

        :return: Uma resposta SAT especializada em ``ConsultarStatusOperacional``.
        :rtype: satcfe.resposta.consultarstatusoperacional.RespostaConsultarStatusOperacional
        """
        resp = self._http_post('consultarstatusoperacional')
        conteudo = resp.json()
        return RespostaConsultarStatusOperacional.analisar(
                conteudo.get('retorno'))