def generate_VD_junction_transfer_matrices(self):
        """Compute the transfer matrices for the VD junction.
        
        Sets the attributes Tvd, Svd, Dvd, lTvd, and lDvd.
        
        """  
        
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}                
        
        #Compute Tvd
        Tvd = {}
        for aa in self.codons_dict.keys():
            current_Tvd = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Tvd[nt2num[codon[2]], nt2num[init_nt]] += self.Rvd[nt2num[codon[2]],nt2num[codon[1]]]*self.Rvd[nt2num[codon[1]],nt2num[codon[0]]] * self.Rvd[nt2num[codon[0]],nt2num[init_nt]]
            Tvd[aa] = current_Tvd
            
        #Compute Svd
        Svd = {}
        for aa in self.codons_dict.keys():
            current_Svd = np.zeros((4, 4))
            for ins_nt in 'ACGT':
                if any([codon.startswith(ins_nt) for codon in self.codons_dict[aa]]):
                    current_Svd[nt2num[ins_nt], :] = self.Rvd[nt2num[ins_nt], :]
                
            Svd[aa] = current_Svd
        
        #Compute Dvd                
        Dvd = {}
        for aa in self.codons_dict.keys():
            current_Dvd = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Dvd[nt2num[codon[2]], nt2num[init_nt]] += self.Rvd[nt2num[codon[1]],nt2num[codon[0]]] * self.Rvd[nt2num[codon[0]],nt2num[init_nt]]
            Dvd[aa] = current_Dvd
     

        #Compute lTvd
        lTvd = {}
        for aa in self.codons_dict.keys():
            current_lTvd = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_lTvd[nt2num[codon[2]], nt2num[codon[0]]] += self.Rvd[nt2num[codon[2]],nt2num[codon[1]]]*self.first_nt_bias_insVD[nt2num[codon[1]]]
            lTvd[aa] = current_lTvd

        
        #Compute lDvd
        lDvd = {}
        for aa in self.codons_dict.keys():
            current_lDvd = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_lDvd[nt2num[codon[2]], nt2num[codon[0]]] += self.first_nt_bias_insVD[nt2num[codon[1]]]
            lDvd[aa] = current_lDvd
        
        #Set the attributes
        self.Tvd = Tvd
        self.Svd = Svd
        self.Dvd = Dvd
        self.lTvd = lTvd
        self.lDvd = lDvd