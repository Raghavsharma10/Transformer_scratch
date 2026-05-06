def _get_notification(self, email, token):
        ''' Consulta o status do pagamento '''        
        url = u'{notification_url}{notification_code}?email={email}&token={token}'.format(
            notification_url=self.notification_url,
            notification_code=self.notification_code,
            email=email,
            token=token)
        req = requests.get(url)
        if req.status_code == 200:
            self.xml = req.text
            logger.debug( u'XML com informacoes da transacao recebido: {0}'.format(self.xml) )
            transaction_dict = xmltodict.parse(self.xml)
            # Validar informações recebidas
            transaction_schema(transaction_dict)
            self.transaction = transaction_dict.get('transaction')
        else:
            raise PagSeguroApiException(
                        u'Erro ao fazer request para a API de notificacao:' + 
                        ' HTTP Status=%s - Response: %s' % (req.status_code, req.text))