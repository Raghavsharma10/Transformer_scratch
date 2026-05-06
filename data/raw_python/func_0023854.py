def execute(self, requests, resp_generator, *args, **kwargs):
        '''
            Calls the resp_generator for all the requests in sequential order.
        '''
        return [resp_generator(request) for request in requests]