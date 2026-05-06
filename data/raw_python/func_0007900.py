def toString(self):
        """ Returns time as string. """
        slist = self.toList()
        string = angle.slistStr(slist)
        return string if slist[0] == '-' else string[1:]