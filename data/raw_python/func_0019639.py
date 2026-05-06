def parse(self, fo, width, seed=None):
        """
        Convert Posmo output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing Posmo output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []

        lines = [fo.readline() for x in range(6)]
        while lines[0]:
            matrix = [[float(x) for x in line.strip().split("\t")] for line in lines[2:]]
            matrix = [[matrix[x][y] for x in range(4)] for y in range(len(matrix[0]))]
            m = Motif(matrix)
            m.trim(0.1)
            m.id = lines[0].strip().split(" ")[-1]
            motifs.append(m)
            lines = [fo.readline() for x in range(6)]
        
        for i,motif in enumerate(motifs):
            if seed:
                motif.id = "%s_w%s.%s_%s" % (self.name, width, seed, i + 1)
            else:
                motif.id = "%s_w%s_%s" % (self.name, width, i + 1)
            motif.trim(0.25)
        
        return motifs