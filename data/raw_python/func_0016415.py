def delete_all_customers(self):
        '''
        This method does exactly what you think it does.  Calling this method
        deletes all customer data in your cheddar product and the configured
        gateway.  This action cannot be undone.
        
        DO NOT RUN THIS UNLESS YOU REALLY, REALLY, REALLY MEAN TO!
        '''
        response = self.client.make_request(
            path='customers/delete-all/confirm/%d' % int(time()),
            method='POST'
        )