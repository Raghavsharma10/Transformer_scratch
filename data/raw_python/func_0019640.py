def parse(self, fo):
        """
        Convert GADEM output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing GADEM output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        nucs = {"A":0,"C":1,"G":2,"T":3}

        lines = fo.readlines()
        for i in range(0, len(lines), 5):
            align = []
            pwm = []
            pfm = []
            m_id = ""
            line = lines[i].strip()
            m_id = line[1:]
            number = m_id.split("_")[0][1:]
            if os.path.exists("%s.seq" % number):
                with open("%s.seq" % number) as f:
                    for l in f:
                        if "x" not in l and "n" not in l:
                            l = l.strip().upper()
                            align.append(l)
                            if not pfm:
                                pfm = [[0 for x in range(4)] for x in range(len(l))]
                            for p in range(len(l)):
                                pfm[p][nucs[l[p]]] += 1
    
            m = [l.strip().split(" ")[1].split("\t") for l in lines[i + 1: i + 5]]

            pwm = [[float(m[x][y]) for x in range(4)] for y in range(len(m[0]))]


            motifs.append(Motif(pwm))
            motifs[-1].id = "{}_{}".format(self.name, m_id)
            #motifs[-1].pwm = pwm
            if align:
                motifs[-1].pfm = pfm
                motifs[-1].align = align

        return motifs