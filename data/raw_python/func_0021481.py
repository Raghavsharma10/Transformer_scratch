def compute_CDR3_pgen(self, CDR3_seq, V_usage_mask, J_usage_mask):
        """Compute Pgen for CDR3 'amino acid' sequence CDR3_seq from VJ model.
    
        Conditioned on the already formatted V genes/alleles indicated in 
        V_usage_mask and the J genes/alleles in J_usage_mask.
    
        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        V_usage_mask : list
            Indices of the V alleles to be considered in the Pgen computation
        J_usage_mask : list
            Indices of the J alleles to be considered in the Pgen computation
    
        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence
        
        Examples
        --------
        >>> compute_CDR3_pgen('CAVKIQGAQKLVF', ppp, [72], [56])
        4.1818202431143785e-07
        >>> compute_CDR3_pgen(nt2codon_rep('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC'), ppp, [42], [1])
        1.3971676613008565e-08
        >>> compute_CDR3_pgen('\xbb\xb6\xbe\x80\xbc\xa1\x8a\x96\xa1\xa0\xad\x8e\xbf', ppp, [72], [56])
        1.3971676613008565e-08
        
        """
        
        #Genomic J alignment/matching (contribution from P(delJ | J)), return Pi_J and reduced J_usage_mask
        Pi_J, r_J_usage_mask = self.compute_Pi_J(CDR3_seq, J_usage_mask)
        
        #Genomic V alignment/matching conditioned on J gene (contribution from P(V, J, delV)), return Pi_V_given_J
        Pi_V_given_J, max_V_align = self.compute_Pi_V_given_J(CDR3_seq, V_usage_mask, r_J_usage_mask)
        
        #Include insertions (R and PinsVJ) to get the total contribution from the left (3') side conditioned on J gene. Return Pi_V_insVJ_given_J
        Pi_V_insVJ_given_J = self.compute_Pi_V_insVJ_given_J(CDR3_seq, Pi_V_given_J, max_V_align)
        
        pgen = 0
        #zip Pi_V_insVJ_given_J and Pi_J together for each J gene to get total pgen
        for j in range(len(r_J_usage_mask)):
            for pos in range(len(CDR3_seq)*3 - 1):
                pgen += np.dot(Pi_V_insVJ_given_J[j][:, pos], Pi_J[j][:, pos+1])
        return pgen