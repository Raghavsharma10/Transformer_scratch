def _additional(self, idents, kwargs):
        '''Add additional data slots from **kwargs'''
        if kwargs:
            for name, value in list(kwargs.items()):
                if not isinstance(value, list):
                    raise ValueError('Additional arguments must be lists of \
same length as idents')
                for i in range(len(value)):
                    self[idents[i]][name] = value[i]