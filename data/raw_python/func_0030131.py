def time_description(self):
        """String description of the year or year range"""

        tc = [t for t in self._p.time_coverage if t]

        if not tc:
            return ''

        mn = min(tc)
        mx = max(tc)

        if not mn and not mx:
            return ''
        elif mn == mx:
            return mn
        else:
            return "{} to {}".format(mn, mx)