def find_ge(self, dt):
        '''Building block of all searches. Find the index
corresponding to the leftmost value greater or equal to *dt*.
If *dt* is greater than the
:func:`dynts.TimeSeries.end` a :class:`dynts.exceptions.RightOutOfBound`
exception will raise.

*dt* must be a python datetime.date instance.'''
        i = bisect_left(self.dates, dt)
        if i != len(self.dates):
            return i
        raise RightOutOfBound