def generate_DJ_junction_transfer_matrices(self):
        """Compute the transfer matrices for the VD junction.
        
        Sets the attributes Tdj, Sdj, Ddj, rTdj, and rDdj.
        
        """   
        
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        #Compute Tdj    
        Tdj = {}
        for aa in self.codons_dict.keys():
            current_Tdj = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Tdj[nt2num[codon[0]], nt2num[init_nt]] += self.Rdj[nt2num[codon[0]],nt2num[codon[1]]]*self.Rdj[nt2num[codon[1]],nt2num[codon[2]]] * self.Rdj[nt2num[codon[2]],nt2num[init_nt]]
            Tdj[aa] = current_Tdj
        
        #Compute Sdj
        Sdj = {}
        for aa in self.codons_dict.keys():
            current_Sdj = np.zeros((4, 4))
            for ins_nt in 'ACGT':
                if any([codon.endswith(ins_nt) for codon in self.codons_dict[aa]]):
                    current_Sdj[nt2num[ins_nt], :] = self.Rdj[nt2num[ins_nt], :]    
            Sdj[aa] = current_Sdj
        
        #Compute Ddj
        Ddj = {}
        for aa in self.codons_dict.keys():
            current_Ddj = np.zeros((4, 4))
            for init_nt in 'ACGT':
                for codon in self.codons_dict[aa]:
                    current_Ddj[nt2num[codon[0]], nt2num[init_nt]] += self.Rdj[nt2num[codon[1]],nt2num[codon[2]]] * self.Rdj[nt2num[codon[2]],nt2num[init_nt]]
            Ddj[aa] = current_Ddj
        
        #Compute rTdj
        rTdj = {}
        for aa in self.codons_dict.keys():
            current_lTdj = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_lTdj[nt2num[codon[0]], nt2num[codon[2]]] += self.Rdj[nt2num[codon[0]],nt2num[codon[1]]]*self.first_nt_bias_insDJ[nt2num[codon[1]]]
            rTdj[aa] = current_lTdj
        
        #Compute rDdj
        rDdj = {}
        for aa in self.codons_dict.keys():
            current_rDdj = np.zeros((4, 4))
            for codon in self.codons_dict[aa]:
                current_rDdj[nt2num[codon[0]], nt2num[codon[2]]] += self.first_nt_bias_insDJ[nt2num[codon[1]]]
            rDdj[aa] = current_rDdj
    
        #Set the attributes
        self.Tdj = Tdj
        self.Sdj = Sdj
        self.Ddj = Ddj
        self.rTdj = rTdj
        self.rDdj = rDdj