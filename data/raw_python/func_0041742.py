def multi_lpop(self, queue, number, transaction=False):
        ''' Pops multiple elements from a list
            This operation will be atomic if transaction=True is passed
        '''
        try:
            pipe = self.pipeline(transaction=transaction)
            pipe.multi()
            self._multi_lpop_pipeline(pipe, queue, number)
            return pipe.execute()[0]
        except IndexError:
            return []
        except:
            raise