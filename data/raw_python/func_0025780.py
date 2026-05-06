def get_current_index(self):
        """ Return currently selected index (or -1) """

        # Need to convert to int; currently API returns a tuple of string
        curSel = self.__lb.curselection()
        if curSel and len(curSel) > 0:
            return int(curSel[0])
        else:
            return -1