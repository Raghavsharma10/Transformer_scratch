def parse_out(self, fo):
        """
        Convert MotifSampler output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing MotifSampler output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        nucs = {"A":0,"C":1,"G":2,"T":3}
        pseudo = 0.0 # Should be 1/sqrt(# of seqs)
        aligns = {}
        for line in fo.readlines():
            if line.startswith("#"):
                pass
            elif len(line) > 1:
                vals = line.strip().split("\t")
                m_id, site = [x.strip().split(" ")[1].replace('"',"") for x in vals[8].split(";") if x]
                #if vals[6] == "+":
                if site.upper().find("N") == -1:
                    aligns.setdefault(m_id, []).append(site)
                #else:
                #    print site, rc(site)
                #    aligns.setdefault(id, []).append(rc(site))
                        
        for m_id, align in aligns.items():
            #print id, len(align)

            width = len(align[0])
            pfm =  [[0 for x in range(4)] for x in range(width)]
            for row in align:
                for i in range(len(row)):
                    pfm[i][nucs[row[i]]] += 1
            total = float(len(align))
            pwm = [[(x + pseudo/4)/total+(pseudo) for x in row] for row in pfm]
            m = Motif()
            m.align = align[:]
            m.pwm = pwm[:]
            m.pfm = pfm[:]
            m.id = m_id
            motifs.append(m)    
        return motifs