def multi_rpush(self, queue, values, bulk_size=0, transaction=False):
        ''' Pushes multiple elements to a list '''
        # Check that what we receive is iterable
        if hasattr(values, '__iter__'):
            self._multi_rpush_pipeline(self, queue, values, 0)
        else:
            raise ValueError('Expected an iterable')