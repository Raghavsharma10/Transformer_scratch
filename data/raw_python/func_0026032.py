def pretty(self,verbose=0):
        """Return pretty list description of parameter"""
        # split prompt lines and add blanks in later lines to align them
        plines = self.prompt.split('\n')
        for i in range(len(plines)-1): plines[i+1] = 32*' ' + plines[i+1]
        plines = '\n'.join(plines)
        namelen = min(len(self.name), 12)
        pvalue = self.get(prompt=0,lpar=1)
        alwaysquoted = ['s', 'f', '*gcur', '*imcur', '*ukey', 'pset']
        if self.type in alwaysquoted and self.value is not None: pvalue = '"' + pvalue + '"'
        if self.mode == "h":
            s = "%13s = %-15s %s" % ("("+self.name[:namelen],
                                    pvalue+")", plines)
        else:
            s = "%13s = %-15s %s" % (self.name[:namelen],
                                    pvalue, plines)
        if not verbose: return s

        if self.choice is not None:
            s = s + "\n" + 32*" " + "|"
            nline = 33
            for i in range(len(self.choice)):
                sch = str(self.choice[i]) + "|"
                s = s + sch
                nline = nline + len(sch) + 1
                if nline > 80:
                    s = s + "\n" + 32*" " + "|"
                    nline = 33
        elif self.min not in [None, INDEF] or self.max not in [None, INDEF]:
            s = s + "\n" + 32*" "
            if self.min not in [None, INDEF]:
                s = s + str(self.min) + " <= "
            s = s + self.name
            if self.max not in [None, INDEF]:
                s = s + " <= " + str(self.max)
        return s