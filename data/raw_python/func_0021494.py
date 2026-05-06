def generate_VJ_junction_transfer_matrices(self):
        """Compute the transfer matrices for the VJ junction.
        
        Sets the attributes Tvj, Svj, Dvj, lTvj, and lDvj.
        
        """    
        
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}                
        
        #Compute Tvj
        Tvj = {}
        for aa in self.codons_dict.keys():
            current_Tvj = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Tvj[nt2num[codon[2]], nt2num[init_nt]] += self.Rvj[nt2num[codon[2]],nt2num[codon[1]]]*self.Rvj[nt2num[codon[1]],nt2num[codon[0]]] * self.Rvj[nt2num[codon[0]],nt2num[init_nt]]
            Tvj[aa] = current_Tvj
    
        #Compute Svj
        Svj = {}
        for aa in self.codons_dict.keys():
            current_Svj = np.zeros((4, 4))
            for ins_nt in 'ACGT':
                if any([codon.startswith(ins_nt) for codon in self.codons_dict[aa]]):
                    current_Svj[nt2num[ins_nt], :] = self.Rvj[nt2num[ins_nt], :]             
            Svj[aa] = current_Svj
        
        #Compute Dvj               
        Dvj = {}    
        for aa in self.codons_dict.keys():
            current_Dvj = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Dvj[nt2num[codon[2]], nt2num[init_nt]] += self.Rvj[nt2num[codon[1]],nt2num[codon[0]]] * self.Rvj[nt2num[codon[0]],nt2num[init_nt]]
            Dvj[aa] = current_Dvj

        #Compute lTvj
        lTvj = {}
        for aa in self.codons_dict.keys():
            current_lTvj = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_lTvj[nt2num[codon[2]], nt2num[codon[0]]] += self.Rvj[nt2num[codon[2]],nt2num[codon[1]]]*self.first_nt_bias_insVJ[nt2num[codon[1]]]
            lTvj[aa] = current_lTvj

        #Compute lDvj        
        lDvj = {}    
        for aa in self.codons_dict.keys():
            current_lDvj = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_lDvj[nt2num[codon[2]], nt2num[codon[0]]] += self.first_nt_bias_insVJ[nt2num[codon[1]]]
            lDvj[aa] = current_lDvj
    

        #Set the attributes
        self.Tvj = Tvj
        self.Svj = Svj
        self.Dvj = Dvj
        self.lTvj = lTvj
        self.lDvj = lDvj