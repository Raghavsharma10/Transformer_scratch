def _multi_rpush_pipeline(self, pipe, queue, values, bulk_size=0):
        ''' Pushes multiple elements to a list in a given pipeline
            If bulk_size is set it will execute the pipeline every bulk_size elements
        '''
        cont = 0
        for value in values:
            pipe.rpush(queue, value)
            if bulk_size != 0 and cont % bulk_size == 0:
                pipe.execute()