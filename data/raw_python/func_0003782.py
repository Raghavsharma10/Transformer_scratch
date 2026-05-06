def read(self, line, f, data):
        """See :meth:`PunchParser.read`"""
        data["energy"] = float(f.readline().split()[1])
        N = len(data["symbols"])
        # if the data are already read before, just overwrite them
        gradient = data.get("gradient")
        if gradient is None:
            gradient = np.zeros((N,3), float)
            data["gradient"] = gradient
        for i in range(N):
            words = f.readline().split()
            gradient[i,0] = float(words[2])
            gradient[i,1] = float(words[3])
            gradient[i,2] = float(words[4])