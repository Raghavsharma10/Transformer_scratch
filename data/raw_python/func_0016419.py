def set(self, quantity):
        '''
        Set the item's quantity to the passed in amount.  If nothing is
        passed in, a quantity of 1 is assumed.  If a decimal value is passsed
        in, it is rounded to the 4th decimal place as that is the level of 
        precision which the Cheddar API accepts.
        '''
        data = {}
        data['quantity'] = self._normalize_quantity(quantity)
         
        response = self.subscription.customer.product.client.make_request(
            path = 'customers/set-item-quantity',
            params = {
                'code': self.subscription.customer.code,
                'itemCode': self.code,
            },
            data = data,
            method = 'POST',
        )
        
        return self.subscription.customer.load_data_from_xml(response.content)