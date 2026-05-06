def compute_Pi_JinsDJ_given_D(self, CDR3_seq, Pi_J_given_D, max_J_align):
        """Compute Pi_JinsDJ conditioned on D.
    
        This function returns the Pi array from the model factors of the J genomic 
        contributions, P(D,J)*P(delJ|J), and the DJ (N2) insertions,
        first_nt_bias_insDJ(n_1)PinsDJ(\ell_{DJ})\prod_{i=2}^{\ell_{DJ}}Rdj(n_i|n_{i-1}) 
        conditioned on D identity. This corresponds to {N^{x_3}}_{x_4}J(D)^{x_4}.
        
        For clarity in parsing the algorithm implementation, we include which 
        instance attributes are used in the method as 'parameters.'
    
        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        Pi_J_given_D : ndarray
            List of (4, 3L) ndarrays corresponding to J(D)^{x_4}.
        max_J_align : int
            Maximum alignment of the CDR3_seq to any genomic J allele allowed by
            J_usage_mask.
            
        self.PinsDJ : ndarray
            Probability distribution of the DJ (N2) insertion sequence length    
        self.first_nt_bias_insDJ : ndarray
            (4,) array of the probability distribution of the indentity of the 
            first nucleotide insertion for the DJ junction.        
        self.zero_nt_bias_insDJ : ndarray
            (4,) array of the probability distribution of the indentity of the 
            the nucleotide BEFORE the DJ insertion. Note, as the Markov model
            at the DJ junction goes 3' to 5' this is the position AFTER the
            insertions reading left to right.
        self.Tdj : dict
            Dictionary of full codon transfer matrices ((4, 4) ndarrays) by 
            'amino acid'.
        self.Sdj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the DJ insertion ending in the first position.
        self.Ddj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the VD insertion ending in the second position.
        self.rTdj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for 
            the DJ insertion starting in the first position.
        self.rDdj : dict
            Dictionary of transfer matrices ((4, 4) ndarrays) by 'amino acid' for
            DJ insertion starting in the first position and ending in the second 
            position of the same codon.
    
        Returns
        -------
        Pi_JinsDJ_given_D : list
            List of (4, 3L) ndarrays corresponding to {N^{x_3}}_{x_4}J(D)^{x_4}.
            
        """
        #max_insertions = 30 #len(PinsVD) - 1 should zeropad the last few spots
        max_insertions = len(self.PinsDJ) - 1
        
        
        Pi_JinsDJ_given_D = [np.zeros((4, len(CDR3_seq)*3)) for i in range(len(Pi_J_given_D))]
        
        for D_in in range(len(Pi_J_given_D)):
            #start position is first nt in a codon
            for init_pos in range(-1, -(max_J_align+1), -3):
                #Zero insertions
                Pi_JinsDJ_given_D[D_in][:, init_pos] += self.PinsDJ[0]*Pi_J_given_D[D_in][:, init_pos]
                
                #One insertion
                Pi_JinsDJ_given_D[D_in][:, init_pos-1] += self.PinsDJ[1]*np.dot(self.rDdj[CDR3_seq[init_pos/3]], Pi_J_given_D[D_in][:, init_pos])
        
                #Two insertions and compute the base nt vec for the standard loop        
                current_base_nt_vec = np.dot(self.rTdj[CDR3_seq[init_pos/3]], Pi_J_given_D[D_in][:, init_pos])
                Pi_JinsDJ_given_D[D_in][0, init_pos-2] += self.PinsDJ[2]*np.sum(current_base_nt_vec)
                
                base_ins = 2
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 - 1: init_pos/3 - max_insertions/3:-1]:
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-1] += self.PinsDJ[base_ins + 1]*np.dot(self.Sdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-2] += self.PinsDJ[base_ins + 2]*np.dot(self.Ddj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][0, init_pos-base_ins-3] += self.PinsDJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
                
            
            #start position is second nt in a codon
            for init_pos in range(-2, -(max_J_align+1), -3):
                #Zero insertions
                Pi_JinsDJ_given_D[D_in][:, init_pos] += self.PinsDJ[0]*Pi_J_given_D[D_in][:, init_pos]
                #One insertion --- we first compute our p vec by pairwise mult with the ss distr
                current_base_nt_vec = np.multiply(Pi_J_given_D[D_in][:, init_pos], self.first_nt_bias_insDJ)
                Pi_JinsDJ_given_D[D_in][0, init_pos-1] += self.PinsDJ[1]*np.sum(current_base_nt_vec)
                base_ins = 1
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 - 1: init_pos/3 - max_insertions/3:-1]:
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-1] += self.PinsDJ[base_ins + 1]*np.dot(self.Sdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-2] += self.PinsDJ[base_ins + 2]*np.dot(self.Ddj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][0, init_pos-base_ins-3] += self.PinsDJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
                
            #start position is last nt in a codon   
            for init_pos in range(-3, -(max_J_align+1), -3):
                
                #Zero insertions
                Pi_JinsDJ_given_D[D_in][0, init_pos] += self.PinsDJ[0]*Pi_J_given_D[D_in][0, init_pos]
                #current_base_nt_vec = first_nt_bias_insDJ*Pi_J_given_D[D_in][0, init_pos] #Okay for steady state
                current_base_nt_vec = self.zero_nt_bias_insDJ*Pi_J_given_D[D_in][0, init_pos]
                base_ins = 0
                
                #Loop over all other insertions using base_nt_vec
                for aa in CDR3_seq[init_pos/3 - 1: init_pos/3 - max_insertions/3:-1]:
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-1] += self.PinsDJ[base_ins + 1]*np.dot(self.Sdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][:, init_pos-base_ins-2] += self.PinsDJ[base_ins + 2]*np.dot(self.Ddj[aa], current_base_nt_vec)
                    current_base_nt_vec = np.dot(self.Tdj[aa], current_base_nt_vec)
                    Pi_JinsDJ_given_D[D_in][0, init_pos-base_ins-3] += self.PinsDJ[base_ins + 3]*np.sum(current_base_nt_vec)
                    base_ins +=3
    
         
        return Pi_JinsDJ_given_D