def time(self):
        """ Returns time as list [hh,mm,ss]. """
        slist = self.toList()
        if slist[0] == '-':
            slist[1] *= -1
            # We must do a trick if we want to 
            # make negative zeros explicit
            if slist[1] == -0:
                slist[1] = -0.0
        return slist[1:]