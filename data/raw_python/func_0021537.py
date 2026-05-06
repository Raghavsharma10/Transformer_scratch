def generate_cutV_genomic_CDR3_segs(self):
        """Add palindromic inserted nucleotides to germline V sequences.
        
        The maximum number of palindromic insertions are appended to the
        germline V segments so that delV can index directly for number of
        nucleotides to delete from a segment.
        
        Sets the attribute cutV_genomic_CDR3_segs.
        
        """
    
        max_palindrome = self.max_delV_palindrome

        self.cutV_genomic_CDR3_segs = []
        for CDR3_V_seg in [x[1] for x in self.genV]:
            if len(CDR3_V_seg) < max_palindrome:
                self.cutV_genomic_CDR3_segs += [cutR_seq(CDR3_V_seg, 0, len(CDR3_V_seg))]
            else:
                self.cutV_genomic_CDR3_segs += [cutR_seq(CDR3_V_seg, 0, max_palindrome)]