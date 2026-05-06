def detalhe(self, posicao=0):
        """Retorna o detalhe de um CEP da última lista de resultados"""
        handle = self._url_open('detalheCEPAction.do', {'Metodo': 'detalhe',
                                                        'TipoCep': 2,
                                                        'Posicao': posicao + 1,
                                                        'CEP': ''})
        html = handle.read()
        return self._parse_detalhe(html)