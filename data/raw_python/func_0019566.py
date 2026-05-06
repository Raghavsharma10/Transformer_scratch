def pwm_scan_to_gff(self, fa, gfffile, cutoff=0.9, nreport=50, scan_rc=True, append=False):
        """Scan sequences with this motif and save to a GFF file.

        Scan sequences from a FASTA object with this motif. Less efficient 
        than using a Scanner object. By setting the cutoff to 0.0 and 
        nreport to 1, the best match for every sequence will be returned.
        The output is save to a file in GFF format.

        Parameters
        ----------
        fa : Fasta object
            Fasta object to scan.
        gfffile : str
            Filename of GFF output file.
        cutoff : float , optional
            Cutoff to use for motif scanning. This cutoff is not specifically
            optimized and the strictness will vary a lot with motif lengh.
        nreport : int , optional
            Maximum number of matches to report.
        scan_rc : bool , optional
            Scan the reverse complement. True by default.
        append : bool , optional
            Append to GFF file instead of overwriting it. False by default.
        """
        if append:
            out = open(gfffile, "a")
        else:    
            out = open(gfffile, "w")

        c = self.pwm_min_score() + (self.pwm_max_score() - self.pwm_min_score()) * cutoff        
        pwm = self.pwm

        strandmap = {-1:"-","-1":"-","-":"-","1":"+",1:"+","+":"+"}
        gff_line = ("{}\tpfmscan\tmisc_feature\t{}\t{}\t{:.3f}\t{}\t.\t"
                    "motif_name \"{}\" ; motif_instance \"{}\"\n")
        for name, seq in fa.items():
            result = pfmscan(seq.upper(), pwm, c, nreport, scan_rc)
            for score, pos, strand in result:
                out.write(gff_line.format( 
                    name, 
                    pos, 
                    pos + len(pwm), 
                    score, 
                    strandmap[strand], 
                    self.id, 
                    seq[pos:pos + len(pwm)]
                    ))
        out.close()