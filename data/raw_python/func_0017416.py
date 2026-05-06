def find_le(self, dt):
        '''Find the index corresponding to the rightmost
value less than or equal to *dt*.
If *dt* is less than :func:`dynts.TimeSeries.end`
a :class:`dynts.exceptions.LeftOutOfBound`
exception will raise.

*dt* must be a python datetime.date instance.'''
        i = bisect_right(self.dates, dt)
        if i:
            return i-1
        raise LeftOutOfBound