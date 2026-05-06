def read(self, line, f, data):
        """See :meth:`PunchParser.read`"""
        self.used = True
        data["title"] = f.readline().strip()
        data["symmetry"] = f.readline().split()[0]
        if data["symmetry"] != "C1":
            raise NotImplementedError("Only C1 symmetry is supported.")
        symbols = []
        while line != " $END      \n":
            line = f.readline()
            if line[0] != " ":
                symbols.append(line.split()[0])
        data["symbols"] = symbols