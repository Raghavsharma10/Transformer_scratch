def charge(self, code, each_amount, quantity=1, description=None):
        '''
        Add an arbitrary charge or credit to a customer's account.  A positive
        number will create a charge.  A negative number will create a credit.
        
        each_amount is normalized to a Decimal with a precision of 2 as that
        is the level of precision which the cheddar API supports.
        '''
        each_amount = Decimal(each_amount)
        each_amount = each_amount.quantize(Decimal('.01'))
        data = {
            'chargeCode': code,
            'eachAmount': '%.2f' % each_amount,
            'quantity': quantity,
        }
        if description:
            data['description'] = description
        
        response = self.product.client.make_request(
            path='customers/add-charge',
            params={'code': self.code},
            data=data,
        )
        return self.load_data_from_xml(response.content)