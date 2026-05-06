def _multi_lpop_pipeline(self, pipe, queue, number):
        ''' Pops multiple elements from a list in a given pipeline'''
        pipe.lrange(queue, 0, number - 1)
        pipe.ltrim(queue, number, -1)