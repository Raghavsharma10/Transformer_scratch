def parse(self, fo):
        """
        Convert BioProspector output to motifs
        
        Parameters
        ----------
        fo : file-like
            File object containing BioProspector output.

        Returns
        -------
        motifs : list
            List of Motif instances.
        """
        motifs = []
        
        p = re.compile(r'^\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')
        pwm = []
        motif_id = ""
        for line in fo.readlines():
            if line.startswith("Motif #"):
                if pwm:
                    m = Motif(pwm)
                    m.id = "BioProspector_w%s_%s" % (len(m), motif_id)
                    motifs.append(m)
                motif_id =  line.split("#")[1].split(":")[0]
                pwm = []
            else:
                m = p.search(line)
                if m:
                    pwm.append([float(m.group(x))/100.0 for x in range(1,5)])

        if pwm:
            m = Motif(pwm)
            m.id = "BioProspector_w%s_%s" % (len(m), motif_id)
            motifs.append(m)
        return motifs