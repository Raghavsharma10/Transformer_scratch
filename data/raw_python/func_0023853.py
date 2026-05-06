def execute(self, requests, resp_generator, *args, **kwargs):
        '''
            Calls the resp_generator for all the requests in parallel in an asynchronous way.
        '''
        result_futures = [self.executor_pool.submit(resp_generator, req, *args, **kwargs) for req in requests]
        resp = [res_future.result() for res_future in result_futures]
        return resp