def dates(self, start, end):
        '''Internal function which perform pre-conditioning on dates:

:keyword start: start date.
:keyword end: end date.

This function makes sure the *start* and *end* date are consistent.
It *never fails* and always return a two-element tuple
containing *start*, *end* with *start* less or equal *end*
and *end* never after today.
There should be no reason to override this function.'''
        td    = date.today()
        end   = safetodate(end) or td
        end   = end if end <= td else td
        start = safetodate(start)
        if not start or start > end:
            start = end - timedelta(days=int(round(30.4*
                                                   settings.months_history)))
        return start,end