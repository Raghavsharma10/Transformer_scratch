def parse(self, fo):
        """
        Convert Improbizer output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing Improbizer output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        p = re.compile(r'\d+\s+@\s+\d+\.\d+\s+sd\s+\d+\.\d+\s+(\w+)$')

        line = fo.readline()
        while line and line.find("Color") == -1:
            m = p.search(line)
            if m:
                pwm_data = {}
                for i in range(4):
                    vals = [x.strip() for x in fo.readline().strip().split(" ") if x]
                    pwm_data[vals[0].upper()] = vals[1:]
                pwm = []
                for i in range(len(pwm_data["A"])):
                    pwm.append([float(pwm_data[x][i]) for x in ["A","C","G","T"]])
                motifs.append(Motif(pwm))
                motifs[-1].id = "%s_%s" % (self.name, m.group(1))
            line = fo.readline()
        
        return motifs