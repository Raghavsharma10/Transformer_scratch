def max_nt_to_aa_alignment_right(self, CDR3_seq, ntseq):
        """Find maximum match between CDR3_seq and ntseq from the right.
    
        This function returns the length of the maximum length nucleotide
        subsequence of ntseq contiguous from the right (or 3' end) that is 
        consistent with the 'amino acid' sequence CDR3_seq
    
        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        ntseq : str
            Genomic (J locus) nucleotide sequence to match. 
    
        Returns
        -------
        max_alignment : int
            Maximum length (in nucleotides) nucleotide sequence that matches the 
            CDR3 'amino acid' sequence.
        
        Example
        --------
        >>> generation_probability.max_nt_to_aa_alignment_right('CASSSEGAGGPSLRGHEQFF', 'TTCATGAACACTGAAGCTTTCTTT')
        6
            
        """
        r_CDR3_seq = CDR3_seq[::-1] #reverse CDR3_seq
        r_ntseq = ntseq[::-1] #reverse ntseq
        max_alignment = 0
        if len(ntseq) == 0:
            return 0
        aa_aligned = True
        while aa_aligned:
            if r_ntseq[max_alignment:max_alignment+3][::-1] in self.codons_dict[r_CDR3_seq[max_alignment/3]]:
                max_alignment += 3
                if max_alignment/3 == len(CDR3_seq):
                    return max_alignment
            else:
                break
                aa_aligned = False
        r_last_codon = r_ntseq[max_alignment:max_alignment+3]
        codon_frag = ''
        for nt in r_last_codon:
            codon_frag = nt + codon_frag
            if codon_frag in self.sub_codons_right[r_CDR3_seq[max_alignment/3]]:
                max_alignment += 1
            else:
                break
        return max_alignment