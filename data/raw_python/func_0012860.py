def initdict(self, fname):
        """create a blank dictionary"""
        if isinstance(fname, Idd):
            self.dt, self.dtls = fname.dt, fname.dtls
            return self.dt, self.dtls

        astr = mylib2.readfile(fname)
        nocom = removecomment(astr, '!')
        idfst = nocom
        alist = idfst.split(';')
        lss = []
        for element in alist:
            lst = element.split(',')
            lss.append(lst)

        for i in range(0, len(lss)):
            for j in range(0, len(lss[i])):
                lss[i][j] = lss[i][j].strip()

        dt = {}
        dtls = []
        for element in lss:
            if element[0] == '':
                continue
            dt[element[0].upper()] = []
            dtls.append(element[0].upper())

        self.dt, self.dtls = dt, dtls
        return dt, dtls