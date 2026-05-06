def get_customers(self, filter_data=None):
        '''
        Returns all customers. Sometimes they are too much and cause internal 
        server errors on CG. API call permits post parameters for filtering 
        which tends to fix this
        https://cheddargetter.com/developers#all-customers

        filter_data
            Will be processed by urlencode and can be used for filtering
            Example value: [
                ("subscriptionStatus": "activeOnly"),
                ("planCode[]": "100GB"), ("planCode[]": "200GB")
            ]
        '''
        customers = []
        
        try:
            response = self.client.make_request(path='customers/get', data=filter_data)
        except NotFound:
            response = None
        
        if response:
            customer_parser = CustomersParser()
            customers_data = customer_parser.parse_xml(response.content)
            for customer_data in customers_data:
                customers.append(Customer(product=self, **customer_data))
            
        return customers