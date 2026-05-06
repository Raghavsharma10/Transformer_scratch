def read(self, line, f, data):
        """See :meth:`PunchParser.read`"""
        line = f.readline()
        assert(line == " $HESS\n")
        while line != " $END\n":
            line = f.readline()