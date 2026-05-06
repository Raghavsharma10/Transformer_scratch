def parse(self, fo):
        """
        Convert MEME output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing MEME output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        nucs = {"A":0,"C":1,"G":2,"T":3}

        p = re.compile('MOTIF.+MEME-(\d+)\s*width\s*=\s*(\d+)\s+sites\s*=\s*(\d+)')
        pa = re.compile('\)\s+([A-Z]+)')
        line = fo.readline()
        while line:
            m = p.search(line)
            align = []
            pfm = None  
            if m:
                #print(m.group(0))
                id = "%s_%s_w%s" % (self.name, m.group(1), m.group(2))
                while not line.startswith("//"):
                    ma = pa.search(line)
                    if ma:
                        #print(ma.group(0))
                        l = ma.group(1)
                        align.append(l)
                        if not pfm:
                            pfm = [[0 for x in range(4)] for x in range(len(l))]
                        for pos in range(len(l)):
                            if l[pos] in nucs:
                                pfm[pos][nucs[l[pos]]] += 1
                            else:
                                for i in range(4):
                                    pfm[pos][i] += 0.25
                    
                    line = fo.readline()
                
                motifs.append(Motif(pfm[:]))
                motifs[-1].id = id
                motifs[-1].align = align[:]
            line = fo.readline()

        return motifs