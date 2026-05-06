def parse(self, fo):
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

        pwm = []
        info = {}
        for line in fo.readlines():
            if line.startswith("#"):
                vals =  line.strip()[1:].split(" = ")
                if len(vals) > 1:
                    info[vals[0]] = vals[1]
            elif len(line) > 1:
                pwm.append([float(x) for x in line.strip().split("\t")])
            else:
                motifs.append(Motif())
                motifs[-1].consensus = info["Consensus"]
                motifs[-1].width = info["W"]
                motifs[-1].id = info["ID"]
                motifs[-1].pwm = pwm[:]
                pwm = []
            
        return motifs