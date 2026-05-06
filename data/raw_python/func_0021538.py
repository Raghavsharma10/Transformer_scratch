def generate_cutJ_genomic_CDR3_segs(self):
        """Add palindromic inserted nucleotides to germline J sequences.
        
        The maximum number of palindromic insertions are appended to the
        germline J segments so that delJ can index directly for number of
        nucleotides to delete from a segment.
        
        Sets the attribute cutJ_genomic_CDR3_segs.
        
        """
        
        max_palindrome = self.max_delJ_palindrome
        self.cutJ_genomic_CDR3_segs = []
        for CDR3_J_seg in [x[1] for x in self.genJ]:
            if len(CDR3_J_seg) < max_palindrome:
                self.cutJ_genomic_CDR3_segs += [cutL_seq(CDR3_J_seg, 0, len(CDR3_J_seg))]
            else:
                self.cutJ_genomic_CDR3_segs += [cutL_seq(CDR3_J_seg, 0, max_palindrome)]