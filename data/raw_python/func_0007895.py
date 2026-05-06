def toList(self):
        """ Returns date as signed list. """
        date = self.date()
        sign = '+' if date[0] >= 0 else '-'
        date[0] = abs(date[0])
        return list(sign) + date