def parse(self, fo):
        """
        Convert ChIPMunk output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing ChIPMunk output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        #KDIC|6.124756232026243
        #A|517.9999999999999 42.99999999999999 345.99999999999994 25.999999999999996 602.9999999999999 155.99999999999997 2.9999999999999996 91.99999999999999
        #C|5.999999999999999 4.999999999999999 2.9999999999999996 956.9999999999999 91.99999999999999 17.999999999999996 22.999999999999996 275.99999999999994
        #G|340.99999999999994 943.9999999999999 630.9999999999999 6.999999999999999 16.999999999999996 48.99999999999999 960.9999999999999 14.999999999999998
        #T|134.99999999999997 7.999999999999999 19.999999999999996 9.999999999999998 287.99999999999994 776.9999999999999 12.999999999999998 616.9999999999999
        #N|999.9999999999998
        line = fo.readline()
        if not line:
            return []
        
        while not line.startswith("A|"):
            line = fo.readline() 
        matrix = []
        for _ in range(4):
            matrix.append([float(x) for x in line.strip().split("|")[1].split(" ")])
            line = fo.readline()
        #print matrix
        matrix = [[matrix[x][y] for x in range(4)] for y in range(len(matrix[0]))]
        #print matrix
        m = Motif(matrix)
        m.id = "ChIPMunk_w%s" % len(m)
        return [m]