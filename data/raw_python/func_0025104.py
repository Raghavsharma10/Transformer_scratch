def comunicar_certificado_icpbrasil(self, certificado):
        """Função ``ComunicarCertificadoICPBRASIL`` conforme ER SAT, item 6.1.2.
        Envio do certificado criado pela ICP-Brasil.

        :param str certificado: Conteúdo do certificado digital criado pela
            autoridade certificadora ICP-Brasil.

        :return: Retorna *verbatim* a resposta da função SAT.
        :rtype: string
        """
        return self.invocar__ComunicarCertificadoICPBRASIL(
                self.gerar_numero_sessao(), self._codigo_ativacao, certificado)