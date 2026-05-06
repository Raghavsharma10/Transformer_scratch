def items(self, desc=None, start_value=None, shift_by=None):
        '''Returns a python ``generator`` which can be used to iterate over
        :func:`dynts.TimeSeries.dates` and :func:`dynts.TimeSeries.values`
        returning a two dimensional
        tuple ``(date,value)`` in each iteration.
        Similar to the python dictionary items
        function.

        :parameter desc: if ``True`` the iteratioon starts from the more
            recent data and proceeds backwards.
        :parameter shift_by: optional parallel shift in values.
        :parameter start_value: optional start value of timeseries.
        '''
        if self:
            if shift_by is None and start_value is not None:
                for cross in self.values():
                    missings = 0
                    if shift_by is None:
                        shift_by = []
                        for v in cross:
                            shift_by.append(start_value - v)
                            if v != v:
                                missings += 1
                    else:
                        for j in range(len(shift_by)):
                            s = shift_by[j]
                            v = cross[j]
                            if s != s:
                                if v == v:
                                    shift_by[j] = start_value - v
                                else:
                                    missings += 1
                    if not missings:
                        break
            if shift_by:
                for d, v in zip(self.dates(desc=desc), self.values(desc=desc)):
                    yield d, v + shift_by
            else:
                for d, v in zip(self.dates(desc=desc), self.values(desc=desc)):
                    yield d, v