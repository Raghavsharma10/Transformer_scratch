def compute_Pi_V_insVJ_given_J(self, CDR3_seq, Pi_V_given_J, max_V_align):
        """Compute Pi_V_insVJ conditioned on J.
    
        This function returns the Pi array from the model factors of the V genomic 
        contributions, P(V, J)*P(delV|V), and the VJ (N) insertions,
        first_nt_bias_insVJ(m_1)PinsVJ(\ell_{VJ})\prod_{i=2}^{\ell_{VJ}}Rvj(m_i|m_{i-1}). 
        This corresponds to V(J)_{x_1}{M^{x_1}}_{x_2}.
        
        For clarity in parsing the algorithm implementation, we include which 
        instance attributes are used in the method as 'parameters.'
    
        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        Pi_V_given_J : ndarray
            List of (4, 3L) ndarrays corresponding to V(J)_{x_1}.
        max_V_align : int
            Maximum alignment of the CDR3_seq to any genomic V allele allowed by
            V_usage_mask.
            
        self.PinsVJ : ndarray
            Probability distribution of the VJ insertion sequence length
        self.first_nt_bias_insVJ : ndarray
            (4,) array of the probability distribution of the indentity of the 
            first nucleotide insertion for the VJ junction.        
        self.zero_nt_bias_insVJ : ndarray
            (4,) array of the probability distribution of the indentity of the 
            the nucleotide BEFORE the VJ insertion.
            zero_nt_bias_insVJ = Rvj^{-1}first_nt_bias_insVJ 
        self.Tvj : dict
            Dictionary of full codon transfer matrices ((4, 4) ndarrays) by 
            'amino acid'.
        self.Svj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion ending in the first position.
        self.Dvj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion ending in the second position.
        self.lTvj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion starting in the first position.
        self.lDvj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for
            VD insertion starting in the first position and ending in the second 
            position of the same codon.
    
        Returns
        -------
        Pi_V_insVJ_given_J : list
            List of (4, 3L) ndarrays corresponding to V(J)_{x_1}{M^{x_1}}_{x_2}.
            
        """
        #max_insertions = 30 #len(PinsVJ) - 1 should zeropad the last few spots
        max_insertions = len(self.PinsVJ) - 1
        
        Pi_V_insVJ_given_J = [np.zeros((4, len(CDR3_seq)*3)) for i in range(len(Pi_V_given_J))]
        
        
        for j in range(len(Pi_V_given_J)):
            #start position is first nt in a codon
            for init_pos in range(0, max_V_align, 3):
                #Zero insertions
                Pi_V_insVJ_given_J[j][:, init_pos] += self.PinsVJ[0]*Pi_V_given_J[j][:, init_pos]
                
                #One insertion
                Pi_V_insVJ_given_J[j][:, init_pos+1] += self.PinsVJ[1]*np.dot(self.lDvj[CDR3_seq[init_pos/3]], Pi_V_given_J[j][:, init_pos])
        
                #Two insertions and compute the base nt vec for the standard loop        
                current_base_nt_vec = np.dot(self.lTvj[CDR3_seq[init_pos/3]], Pi_V_given_J[j][:, init_pos])
                Pi_V_insVJ_given_J[j][0, init_pos+2] += self.PinsVJ[2]*np.sum(current_base_nt_vec)
                
                base_ins = 2
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+1] += self.PinsVJ[base_ins + 1]*np.dot(self.Svj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+2] += self.PinsVJ[base_ins + 2]*np.dot(self.Dvj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tvj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][0, init_pos+base_ins+3] += self.PinsVJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
                
            
            #start position is second nt in a codon
            for init_pos in range(1, max_V_align, 3):
                #Zero insertions
                Pi_V_insVJ_given_J[j][:, init_pos] += self.PinsVJ[0]*Pi_V_given_J[j][:, init_pos]
                #One insertion --- we first compute our p vec by pairwise mult with the ss distr
                current_base_nt_vec = np.multiply(Pi_V_given_J[j][:, init_pos], self.first_nt_bias_insVJ)
                Pi_V_insVJ_given_J[j][0, init_pos+1] += self.PinsVJ[1]*np.sum(current_base_nt_vec)
                base_ins = 1
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+1] += self.PinsVJ[base_ins + 1]*np.dot(self.Svj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+2] += self.PinsVJ[base_ins + 2]*np.dot(self.Dvj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tvj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][0, init_pos+base_ins+3] += self.PinsVJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
                
            #start position is last nt in a codon   
            for init_pos in range(2, max_V_align, 3):
                
                #Zero insertions
                Pi_V_insVJ_given_J[j][0, init_pos] += self.PinsVJ[0]*Pi_V_given_J[j][0, init_pos]
                #current_base_nt_vec = first_nt_bias_insVJ*Pi_V_given_J[j][0, init_pos] #Okay for steady state
                current_base_nt_vec = self.zero_nt_bias_insVJ*Pi_V_given_J[j][0, init_pos]
                base_ins = 0
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 + 1: init_pos/3 + max_insertions/3]:
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+1] += self.PinsVJ[base_ins + 1]*np.dot(self.Svj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][:, init_pos+base_ins+2] += self.PinsVJ[base_ins + 2]*np.dot(self.Dvj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tvj[aa], current_base_nt_vec)
                    Pi_V_insVJ_given_J[j][0, init_pos+base_ins+3] += self.PinsVJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
    
         
        return Pi_V_insVJ_given_J