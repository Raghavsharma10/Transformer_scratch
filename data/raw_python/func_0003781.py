def read(self, line, f, data):
        """See :meth:`PunchParser.read`"""
        f.readline()
        f.readline()
        N = len(data["symbols"])
        # if the data are already read before, just overwrite them
        numbers = data.get("numbers")
        if numbers is None:
            numbers = np.zeros(N, int)
            data["numbers"] = numbers
        coordinates = data.get("coordinates")
        if coordinates is None:
            coordinates = np.zeros((N,3), float)
            data["coordinates"] = coordinates
        for i in range(N):
            words = f.readline().split()
            numbers[i] = int(float(words[1]))
            coordinates[i,0] = float(words[2])*angstrom
            coordinates[i,1] = float(words[3])*angstrom
            coordinates[i,2] = float(words[4])*angstrom