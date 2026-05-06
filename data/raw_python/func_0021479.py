def compute_Pi_L(self, CDR3_seq, Pi_V, max_V_align):
        """Compute Pi_L.
    
        This function returns the Pi array from the model factors of the V genomic 
        contributions, P(V)*P(delV|V), and the VD (N1) insertions,
        first_nt_bias_insVD(m_1)PinsVD(\ell_{VD})\prod_{i=2}^{\ell_{VD}}Rvd(m_i|m_{i-1}). 
        This corresponds to V_{x_1}{M^{x_1}}_{x_2}.
        
        For clarity in parsing the algorithm implementation, we include which 
        instance attributes are used in the method as 'parameters.'
    
        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        Pi_V : ndarray
            (4, 3L) array corresponding to V_{x_1}.
        max_V_align : int
            Maximum alignment of the CDR3_seq to any genomic V allele allowed by
            V_usage_mask.
            
        self.PinsVD : ndarray
            Probability distribution of the VD (N1) insertion sequence length            
        self.first_nt_bias_insVD : ndarray
            (4,) array of the probability distribution of the indentity of the 
            first nucleotide insertion for the VD junction.        
        self.zero_nt_bias_insVD : ndarray
            (4,) array of the probability distribution of the indentity of the 
            the nucleotide BEFORE the VD insertion.
            zero_nt_bias_insVD = Rvd^{-1}first_nt_bias_insVD       
        self.Tvd : dict
            Dictionary of full codon transfer matrices ((4, 4) ndarrays) by 
            'amino acid'.
        self.Svd : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion ending in the first position.
        self.Dvd : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion ending in the second position.
        self.lTvd : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion starting in the first position.
        self.lDvd : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for
            VD insertion starting in the first position and ending in the second 
            position of the same codon.
    
        Returns
        -------
        Pi_L : ndarray
            (4, 3L) array corresponding to V_{x_1}{M^{x_1}}_{x_2}.
            
        """
        #max_insertions = 30 #len(PinsVD) - 1 should zeropad the last few spots
        max_insertions = len(self.PinsVD) - 1
        
        Pi_L = np.zeros((4, len(CDR3_seq)*3))
        
        #start position is first nt in a codon
        for init_pos in range(0, max_V_align, 3):
            #Zero insertions
            Pi_L[:, init_pos] += self.PinsVD[0]*Pi_V[:, init_pos]
            
            #One insertion
            Pi_L[:, init_pos+1] += self.PinsVD[1]*np.dot(self.lDvd[CDR3_seq[init_pos/3]], Pi_V[:, init_pos])
    
            #Two insertions and compute the base nt vec for the standard loop        
            current_base_nt_vec = np.dot(self.lTvd[CDR3_seq[init_pos/3]], Pi_V[:, init_pos])
            Pi_L[0, init_pos+2] += self.PinsVD[2]*np.sum(current_base_nt_vec)
            
            base_ins = 2
            
            #Loop over all other insertions using base_nt_vec
            for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                Pi_L[:, init_pos+base_ins+1] += self.PinsVD[base_ins + 1]*np.dot(self.Svd[aa], current_base_nt_vec)
                Pi_L[:, init_pos+base_ins+2] += self.PinsVD[base_ins + 2]*np.dot(self.Dvd[aa], current_base_nt_vec)
                current_base_nt_vec = np.dot(self.Tvd[aa], current_base_nt_vec)
                Pi_L[0, init_pos+base_ins+3] += self.PinsVD[base_ins + 3]*np.sum(current_base_nt_vec)
                base_ins +=3
            
        
        #start position is second nt in a codon
        for init_pos in range(1, max_V_align, 3):
            #Zero insertions
            Pi_L[:, init_pos] += self.PinsVD[0]*Pi_V[:, init_pos]
            #One insertion --- we first compute our p vec by pairwise mult with the ss distr
            current_base_nt_vec = np.multiply(Pi_V[:, init_pos], self.first_nt_bias_insVD)
            Pi_L[0, init_pos+1] += self.PinsVD[1]*np.sum(current_base_nt_vec)
            base_ins = 1
            
            #Loop over all other insertions using base_nt_vec
            for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                Pi_L[:, init_pos+base_ins+1] += self.PinsVD[base_ins + 1]*np.dot(self.Svd[aa], current_base_nt_vec)
                Pi_L[:, init_pos+base_ins+2] += self.PinsVD[base_ins + 2]*np.dot(self.Dvd[aa], current_base_nt_vec)
                current_base_nt_vec = np.dot(self.Tvd[aa], current_base_nt_vec)
                Pi_L[0, init_pos+base_ins+3] += self.PinsVD[base_ins + 3]*np.sum(current_base_nt_vec)
                base_ins +=3
            
        #start position is last nt in a codon   
        for init_pos in range(2, max_V_align, 3):
            
            #Zero insertions
            Pi_L[0, init_pos] += self.PinsVD[0]*Pi_V[0, init_pos]
            #current_base_nt_vec = first_nt_bias_insVD*Pi_V[0, init_pos] #Okay for steady state
            current_base_nt_vec = self.zero_nt_bias_insVD*Pi_V[0, init_pos]
            base_ins = 0
            
            #Loop over all other insertions using base_nt_vec
            for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                Pi_L[:, init_pos+base_ins+1] += self.PinsVD[base_ins + 1]*np.dot(self.Svd[aa], current_base_nt_vec)
                Pi_L[:, init_pos+base_ins+2] += self.PinsVD[base_ins + 2]*np.dot(self.Dvd[aa], current_base_nt_vec)
                current_base_nt_vec = np.dot(self.Tvd[aa], current_base_nt_vec)
                Pi_L[0, init_pos+base_ins+3] += self.PinsVD[base_ins + 3]*np.sum(current_base_nt_vec)
                base_ins +=3
    
         
        return Pi_L