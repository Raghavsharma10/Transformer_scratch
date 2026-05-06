def ativar_sat(self, tipo_certificado, cnpj, codigo_uf):
        """Função ``AtivarSAT`` conforme ER SAT, item 6.1.1.
        Ativação do equipamento SAT. Dependendo do tipo do certificado, o
        procedimento de ativação é complementado enviando-se o certificado
        emitido pela ICP-Brasil (:meth:`comunicar_certificado_icpbrasil`).

        :param int tipo_certificado: Deverá ser um dos valores
            :attr:`satcomum.constantes.CERTIFICADO_ACSAT_SEFAZ`,
            :attr:`satcomum.constantes.CERTIFICADO_ICPBRASIL` ou
            :attr:`satcomum.constantes.CERTIFICADO_ICPBRASIL_RENOVACAO`, mas
            nenhuma validação será realizada antes que a função de ativação
            seja efetivamente invocada.

        :param str cnpj: Número do CNPJ do estabelecimento contribuinte,
            contendo apenas os dígitos. Nenhuma validação do número do CNPJ
            será realizada antes que a função de ativação seja efetivamente
            invocada.

        :param int codigo_uf: Código da unidade federativa onde o equipamento
            SAT será ativado (eg. ``35`` para o Estado de São Paulo). Nenhuma
            validação do código da UF será realizada antes que a função de
            ativação seja efetivamente invocada.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        return self.invocar__AtivarSAT(
                self.gerar_numero_sessao(), tipo_certificado,
                self._codigo_ativacao, cnpj, codigo_uf)