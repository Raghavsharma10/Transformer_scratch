def index(self, date):
        """ Returns the index of a date in the table. """
        for (i, (start, end, ruler)) in enumerate(self.table):
            if start <= date.jd <= end:
                return i
        return None