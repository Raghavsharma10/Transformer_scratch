def consultar_numero_sessao(self, numero_sessao):
        """Função ``ConsultarNumeroSessao`` conforme ER SAT, item 6.1.8.
        Consulta o equipamento SAT por um número de sessão específico.

        :param int numero_sessao: Número da sessão que se quer consultar.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        return self.invocar__ConsultarNumeroSessao(self.gerar_numero_sessao(),
                self._codigo_ativacao, numero_sessao)