def consulta(self, endereco, primeiro=False,
                 uf=None, localidade=None, tipo=None, numero=None):
        """Consulta site e retorna lista de resultados"""

        if uf is None:
            url = 'consultaEnderecoAction.do'
            data = {
                'relaxation': endereco.encode('ISO-8859-1'),
                'TipoCep': 'ALL',
                'semelhante': 'N',
                'cfm': 1,
                'Metodo': 'listaLogradouro',
                'TipoConsulta': 'relaxation',
                'StartRow': '1',
                'EndRow': '10'
            }
        else:
            url = 'consultaLogradouroAction.do'
            data = {
                'Logradouro': endereco.encode('ISO-8859-1'),
                'UF': uf,
                'TIPO': tipo,
                'Localidade': localidade.encode('ISO-8859-1'),
                'Numero': numero,
                'cfm': 1,
                'Metodo': 'listaLogradouro',
                'TipoConsulta': 'logradouro',
                'StartRow': '1',
                'EndRow': '10'
            }

        h = self._url_open(url, data)
        html = h.read()

        if primeiro:
            return self.detalhe()
        else:
            return self._parse_tabela(html)