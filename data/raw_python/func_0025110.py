def trocar_codigo_de_ativacao(self, novo_codigo_ativacao,
            opcao=constantes.CODIGO_ATIVACAO_REGULAR,
            codigo_emergencia=None):
        """Função ``TrocarCodigoDeAtivacao`` conforme ER SAT, item 6.1.15.
        Troca do código de ativação do equipamento SAT.

        :param str novo_codigo_ativacao: O novo código de ativação escolhido
            pelo contribuinte.

        :param int opcao: Indica se deverá ser utilizado o código de ativação
            atualmente configurado, que é um código de ativação regular,
            definido pelo contribuinte, ou se deverá ser usado um código de
            emergência. Deverá ser o valor de uma das constantes
            :attr:`satcomum.constantes.CODIGO_ATIVACAO_REGULAR` (padrão) ou
            :attr:`satcomum.constantes.CODIGO_ATIVACAO_EMERGENCIA`.
            Nenhuma validação será realizada antes que a função seja
            efetivamente invocada. Entretanto, se opção de código de ativação
            indicada for ``CODIGO_ATIVACAO_EMERGENCIA``, então o argumento que
            informa o ``codigo_emergencia`` será checado e deverá avaliar como
            verdadeiro.

        :param str codigo_emergencia: O código de ativação de emergência, que
            é definido pelo fabricante do equipamento SAT. Este código deverá
            ser usado quando o usuário perder o código de ativação regular, e
            precisar definir um novo código de ativação. Note que, o argumento
            ``opcao`` deverá ser informado com o valor
            :attr:`satcomum.constantes.CODIGO_ATIVACAO_EMERGENCIA` para que
            este código de emergência seja considerado.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string

        :raises ValueError: Se o novo código de ativação avaliar como falso
            (possuir uma string nula por exemplo) ou se o código de emergencia
            avaliar como falso quando a opção for pelo código de ativação de
            emergência.

        .. warning::

            Os argumentos da função ``TrocarCodigoDeAtivacao`` requerem que o
            novo código de ativação seja especificado duas vezes (dois
            argumentos com o mesmo conteúdo, como confirmação). Este método irá
            simplesmente informar duas vezes o argumento
            ``novo_codigo_ativacao`` na função SAT, mantendo a confirmação do
            código de ativação fora do escopo desta API.

        """
        if not novo_codigo_ativacao:
            raise ValueError('Novo codigo de ativacao invalido: {!r}'.format(
                    novo_codigo_ativacao))

        codigo_ativacao = self._codigo_ativacao

        if opcao == constantes.CODIGO_ATIVACAO_EMERGENCIA:
            if codigo_emergencia:
                codigo_ativacao = codigo_emergencia
            else:
                raise ValueError('Codigo de ativacao de emergencia invalido: '
                        '{!r} (opcao={!r})'.format(codigo_emergencia, opcao))

        return self.invocar__TrocarCodigoDeAtivacao(
                self.gerar_numero_sessao(), codigo_ativacao, opcao,
                novo_codigo_ativacao, novo_codigo_ativacao)