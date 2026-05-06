def parse(self, fo):
        """
        Convert HMS output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing HMS output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        m = [[float(x) for x in fo.readline().strip().split(" ")] for i in range(4)]
        matrix = [[m[0][i], m[1][i],m[2][i],m[3][i]] for i in range(len(m[0]))]
        motifs = [Motif(matrix)]
        motifs[-1].id = self.name
        
        return motifs