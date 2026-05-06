def analisar(retorno):
        """Constrói uma :class:`RespostaSAT` ou especialização dependendo da
        função SAT encontrada na sessão consultada.

        :param unicode retorno: Retorno da função ``ConsultarNumeroSessao``.
        """
        if '|' not in retorno:
            raise ErroRespostaSATInvalida('Resposta nao possui pipes '
                    'separando os campos: {!r}'.format(retorno))

        resposta = _RespostaParcial(*(retorno.split('|')[:2]))

        for faixa, construtor in _RESPOSTAS_POSSIVEIS:
            if int(resposta.EEEEE) in xrange(faixa, faixa+1000):
                return construtor(retorno)

        return RespostaConsultarNumeroSessao._pos_analise(retorno)