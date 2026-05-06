def multi_lpop(self, queue, number, transaction=False):
        ''' Pops multiple elements from a list '''
        try:
            self._multi_lpop_pipeline(self, queue, number)
        except:
            raise