def create_one_time_invoice(self, charges):
        '''
        Charges should be a list of charges to execute immediately.  Each
        value in the charges diectionary should be a dictionary with the 
        following keys:

        code
            Your code for this charge.  This code will be displayed in the 
            user's invoice and is limited to 36 characters.
        quantity
            A positive integer quantity.  If not provided this value will 
            default to 1.
        each_amount
            Positive or negative integer or decimal with two digit precision.
            A positive number will create a charge (debit). A negative number
            will create a credit.
        description
            An optional description for this charge which will be displayed on
            the user's invoice.
        '''
        data = {}
        for n, charge in enumerate(charges):
            each_amount = Decimal(charge['each_amount'])
            each_amount = each_amount.quantize(Decimal('.01'))
            data['charges[%d][chargeCode]' % n ] = charge['code']
            data['charges[%d][quantity]' % n] = charge.get('quantity', 1)
            data['charges[%d][eachAmount]' % n] = '%.2f' % each_amount
            if 'description' in charge.keys():
                data['charges[%d][description]' % n] = charge['description']

        response = self.product.client.make_request(
            path='invoices/new',
            params={'code': self.code},
            data=data,
        )
        return self.load_data_from_xml(response.content)