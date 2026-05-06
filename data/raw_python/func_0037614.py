def _build_params(self):
        ''' método que constrói o dicionario com os parametros que serão usados
        na requisição HTTP Post ao PagSeguro
        
        Returns:
            Um dicionário com os parametros definidos no objeto Payment.
        '''
        params = {}
        params['email'] = self.email
        params['token'] = self.token
        params['currency'] = self.currency

        # Atributos opcionais
        if self.receiver_email:
            params['receiver_email'] = self.receiver_email
        if self.reference:
            params['reference'] = self.reference
        if self.extra_amount:
            params['extra_amount'] = self.extra_amount
        if self.redirect_url:
            params['redirect_url'] = self.redirect_url
        if self.notification_url:
            params['notification_url'] = self.notification_url
        if self.max_uses:
            params['max_uses'] = self.max_uses
        if self.max_age:
            params['max_age'] = self.max_age

        #TODO: Incluir metadata aqui

        # Itens
        for index, item in enumerate(self.items, start=1):
            params['itemId%d' % index] = item['item_id']
            params['itemDescription%d' % index] = item['description']
            params['itemAmount%d' % index] = '%.2f' % item['amount']
            params['itemQuantity%s' % index] = item['quantity']
            if item.get('shipping_cost'):
                params['itemShippingCost%d' % index] = item['shipping_cost']
            if item.get('weight'):
                params['itemWeight%d' % index] = item['weight']

        # Sender
        if self.client.get('email'):
            params['senderEmail'] = self.client.get('email')
        if self.client.get('name'):
            params['senderName'] = ' '.join(self.client.get('name').split())
        if self.client.get('phone_area_code'):
            params['senderAreaCode'] = self.client.get('phone_area_code')
        if self.client.get('phone_number'):
            params['senderPhone'] = self.client.get('phone_number')
        if self.client.get('cpf'):
            params['senderCPF'] = self.client.get('cpf')
        if self.client.get('sender_born_date'):
            params['senderBornDate'] = self.client.get('sender_born_date')

        # Shipping
        if self.shipping.get('type'):
            params['shippingType'] = self.shipping.get('type')
        if self.shipping.get('cost'):
            params['shippingCost'] = '%.2f' % self.shipping.get('cost')
        if self.shipping.get('country'):
            params['shippingAddressCountry'] = self.shipping.get('country')
        if self.shipping.get('state'):
            params['shippingAddressState'] = self.shipping.get('state')
        if self.shipping.get('city'):
            params['shippingAddressCity'] = self.shipping.get('city')
        if self.shipping.get('postal_code'):
            params['shippingAddressPostalCode'] = self.shipping.get('postal_code')
        if self.shipping.get('district'):
            params['shippingAddressDistrict'] = self.shipping.get('district')
        if self.shipping.get('street'):
            params['shippingAddressStreet'] = self.shipping.get('street')
        if self.shipping.get('number'):
            params['shippingAddressNumber'] = self.shipping.get('number')
        if self.shipping.get('complement'):
            params['shippingAddressComplement'] = self.shipping.get('complement')

        return params