def generate_cutD_genomic_CDR3_segs(self):
        """Add palindromic inserted nucleotides to germline V sequences.
        
        The maximum number of palindromic insertions are appended to the
        germline D segments so that delDl and delDr can index directly for number 
        of nucleotides to delete from a segment.
        
        Sets the attribute cutV_genomic_CDR3_segs.
        
        """
        max_palindrome_L = self.max_delDl_palindrome
        max_palindrome_R = self.max_delDr_palindrome

        self.cutD_genomic_CDR3_segs = []
        for CDR3_D_seg in [x[1] for x in self.genD]:
            if len(CDR3_D_seg) < min(max_palindrome_L, max_palindrome_R):
                self.cutD_genomic_CDR3_segs += [cutR_seq(cutL_seq(CDR3_D_seg, 0, len(CDR3_D_seg)), 0, len(CDR3_D_seg))]
            else:
                self.cutD_genomic_CDR3_segs += [cutR_seq(cutL_seq(CDR3_D_seg, 0, max_palindrome_L), 0, max_palindrome_R)]