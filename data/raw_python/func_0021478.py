def compute_CDR3_pgen(self, CDR3_seq, V_usage_mask, J_usage_mask):
        """Compute Pgen for CDR3 'amino acid' sequence CDR3_seq from VDJ model.
    
        Conditioned on the already formatted V genes/alleles indicated in 
        V_usage_mask and the J genes/alleles in J_usage_mask. 
        (Examples are TCRB sequences/model)
    
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
        >>> compute_CDR3_pgen('CAWSVAPDRGGYTF', ppp, [42], [1])
        1.203646865765782e-10
        >>> compute_CDR3_pgen(nt2codon_rep('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC'), ppp, [42], [1])
        2.3986503758867323e-12
        >>> compute_CDR3_pgen('\xbb\x96\xab\xb8\x8e\xb6\xa5\x92\xa8\xba\x9a\x93\x94\x9f', ppp, [42], [1])
        2.3986503758867323e-12
        
        """
        
        
        #Genomic V alignment/matching (contribution from P(V, delV)), return Pi_V
        Pi_V, max_V_align = self.compute_Pi_V(CDR3_seq, V_usage_mask)
        
        #Include VD insertions (Rvd and PinsVD) to get the total contribution from the left (3') side. Return Pi_L
        Pi_L = self.compute_Pi_L(CDR3_seq, Pi_V, max_V_align)
        
        #Genomic J alignment/matching (contribution from P(D, J, delJ)), return Pi_J_given_D
        Pi_J_given_D, max_J_align = self.compute_Pi_J_given_D(CDR3_seq, J_usage_mask)
        
        #Include DJ insertions (Rdj and PinsDJ), return Pi_JinsDJ_given_D
        Pi_JinsDJ_given_D = self.compute_Pi_JinsDJ_given_D(CDR3_seq, Pi_J_given_D, max_J_align)
        
        #Include D genomic contribution (P(delDl, delDr | D)) to complete the contribution from the right (5') side. Return Pi_R
        Pi_R = self.compute_Pi_R(CDR3_seq, Pi_JinsDJ_given_D)
        
        pgen = 0
        
        #zip Pi_L and Pi_R together to get total pgen
        for pos in range(len(CDR3_seq)*3 - 1):
            pgen += np.dot(Pi_L[:, pos], Pi_R[:, pos+1])
            
        return pgen