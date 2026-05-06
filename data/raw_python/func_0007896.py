def toString(self):
        """ Returns date as string. """
        slist = self.toList()
        sign = '' if slist[0] == '+' else '-'
        string = '/'.join(['%02d' % v for v in slist[1:]])
        return sign + string