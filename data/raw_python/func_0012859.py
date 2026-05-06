def initdict2(self, dictfile):
        """initdict2"""
        dt = {}
        dtls = []
        adict = dictfile
        for element in adict:
            dt[element[0].upper()] = []  # dict keys for objects always in caps
            dtls.append(element[0].upper())
        return dt, dtls