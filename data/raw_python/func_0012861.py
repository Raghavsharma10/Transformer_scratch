def makedict(self, dictfile, fnamefobject):
        """stuff file data into the blank dictionary"""
        #fname = './exapmlefiles/5ZoneDD.idf'
        #fname = './1ZoneUncontrolled.idf'
        if isinstance(dictfile, Idd):
            localidd = copy.deepcopy(dictfile)
            dt, dtls = localidd.dt, localidd.dtls
        else:
            dt, dtls = self.initdict(dictfile)
        # astr = mylib2.readfile(fname)
        astr = fnamefobject.read()
        try:
            astr = astr.decode('ISO-8859-2')
        except AttributeError:
            pass
        fnamefobject.close()
        nocom = removecomment(astr, '!')
        idfst = nocom
        # alist = string.split(idfst, ';')
        alist = idfst.split(';')
        lss = []
        for element in alist:
            # lst = string.split(element, ',')
            lst = element.split(',')
            lss.append(lst)

        for i in range(0, len(lss)):
            for j in range(0, len(lss[i])):
                lss[i][j] = lss[i][j].strip()

        for element in lss:
            node = element[0].upper()
            if node in dt:
                # stuff data in this key
                dt[node.upper()].append(element)
            else:
                # scream
                if node == '':
                    continue
                print('this node -%s-is not present in base dictionary' %
                      (node))

        self.dt, self.dtls = dt, dtls
        return dt, dtls