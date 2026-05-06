def read(self, line, f, data):
        """See :meth:`PunchParser.read`"""
        N = len(data["symbols"])
        masses = np.zeros(N, float)
        counter = 0
        while counter < N:
            words = f.readline().split()
            for word in words:
                masses[counter] = float(word)*amu
                counter += 1
        data["masses"] = masses