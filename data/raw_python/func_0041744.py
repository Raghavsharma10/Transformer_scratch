def multi_rpush(self, queue, values, bulk_size=0, transaction=False):
        ''' Pushes multiple elements to a list
            If bulk_size is set it will execute the pipeline every bulk_size elements
            This operation will be atomic if transaction=True is passed
        '''
        # Check that what we receive is iterable
        if hasattr(values, '__iter__'):
            pipe = self.pipeline(transaction=transaction)
            pipe.multi()
            self._multi_rpush_pipeline(pipe, queue, values, bulk_size)
            pipe.execute()
        else:
            raise ValueError('Expected an iterable')