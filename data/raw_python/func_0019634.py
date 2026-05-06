def parse(self, fo):
        """
        Convert MDmodule output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing MDmodule output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        nucs = {"A":0,"C":1,"G":2,"T":3}
        p = re.compile(r'(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')
        pf = re.compile(r'>.+\s+[bf]\d+\s+(\w+)')

        pwm = []
        pfm = []
        align = []
        m_id = ""
        for line in fo.readlines():
            if line.startswith("Motif"):
                if m_id:
                    motifs.append(Motif())
                    motifs[-1].id = m_id
                    motifs[-1].pwm = pwm
                    motifs[-1].pfm = pfm
                    motifs[-1].align = align
                    pwm = []
                    pfm = []
                    align = []
                m_id = line.split("\t")[0]
            else: 
                m = p.search(line)
                if m:
                    pwm.append([float(m.group(x))/100 for x in [2,3,4,5]])
                m = pf.search(line)
                if m:
                    if not pfm:
                        pfm = [[0 for x in range(4)] for x in range(len(m.group(1)))]
                    for i in range(len(m.group(1))):
                        pfm[i][nucs[m.group(1)[i]]] += 1
                            
                    align.append(m.group(1))
        
        if pwm:
            motifs.append(Motif())
            motifs[-1].id = m_id
            motifs[-1].pwm = pwm
            motifs[-1].pfm = pfm
            motifs[-1].align = align

        return motifs