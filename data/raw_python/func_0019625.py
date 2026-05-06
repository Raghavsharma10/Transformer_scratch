def parse(self, fo):
        """
        Convert AMD output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing AMD output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        
        #160:  112  CACGTGC      7.25   chr14:32308489-32308689
        p = re.compile(r'\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
        wm = []
        name = ""
        for line in fo.readlines():
            if line.startswith("Motif") and line.strip().endswith(":"):
                if name:
                    motifs.append(Motif(wm))
                    motifs[-1].id = name
                    name = ""
                    wm = []
                name = "%s_%s" % (self.name, line.split(":")[0])
            else:
                m = p.search(line)
                if m:
                    wm.append([float(m.group(x)) for x in range(1,5)])
        motifs.append(Motif(wm))
        motifs[-1].id = name
        
        return motifs