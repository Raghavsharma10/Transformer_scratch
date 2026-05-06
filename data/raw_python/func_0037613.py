def request(self):
        '''
        Faz a requisição de pagamento ao servidor do PagSeguro.
        '''
#        try:
        payment_v2_schema(self)
#        except MultipleInvalid as e:
#            raise PagSeguroPaymentValidationException(u'Erro na validação dos dados: %s' % e.msg)
        params = self._build_params()
#         logger.debug(u'Parametros da requisicao ao PagSeguro: %s' % params)
        req = requests.post(
            self.PAGSEGURO_API_URL,
            params=params,
            headers={
                'Content-Type':
                'application/x-www-form-urlencoded; charset=ISO-8859-1'
            }
        )
        if req.status_code == 200:
            self.params = params
            self.response = self._process_response_xml(req.text)
        else:
            raise PagSeguroApiException(
                u'Erro ao fazer request para a API:' +
                ' HTTP Status=%s - Response: %s' % (req.status_code, req.text))
        return