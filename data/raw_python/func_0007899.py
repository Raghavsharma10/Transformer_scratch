def toList(self):
        """ Returns time as signed list. """
        slist = angle.toList(self.value)
        # Keep hours in 0..23
        slist[1] = slist[1] % 24
        return slist