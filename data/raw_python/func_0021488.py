def preprocess_D_segs(self, generative_model, genomic_data):
        """Process P(delDl, delDr|D) into Pi arrays.
        
        Sets the attributes PD_nt_pos_vec, PD_2nd_nt_pos_per_aa_vec, 
        min_delDl_given_DdelDr, max_delDl_given_DdelDr, and zeroD_given_D.
    
        Parameters
        ----------
        generative_model : GenerativeModelVDJ
            VDJ generative model class containing the model parameters.            
        genomic_data : GenomicDataVDJ
            VDJ genomic data class containing the V, D, and J germline 
            sequences and info.
        
        """
    
        cutD_genomic_CDR3_segs = genomic_data.cutD_genomic_CDR3_segs
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num_dell_pos, num_delr_pos, num_D_genes = generative_model.PdelDldelDr_given_D.shape
        
        #These arrays only include the nt identity information, not the PdelDldelDr_given_D info
        PD_nt_pos_vec = [[]]*num_D_genes
        PD_2nd_nt_pos_per_aa_vec = [[]]*num_D_genes
        for D_in in range(num_D_genes):
           
            current_PD_nt_pos_vec = np.zeros((4, len(cutD_genomic_CDR3_segs[D_in])))
            current_PD_2nd_nt_pos_per_aa_vec = {}
            for aa in self.codons_dict.keys():
                current_PD_2nd_nt_pos_per_aa_vec[aa] = np.zeros((4, len(cutD_genomic_CDR3_segs[D_in])))
            
            for pos, nt in enumerate(cutD_genomic_CDR3_segs[D_in]):
                current_PD_nt_pos_vec[nt2num[nt], pos] = 1
                for ins_nt in 'ACGT':
                    for aa in self.codons_dict.keys():
                        if ins_nt + cutD_genomic_CDR3_segs[D_in][pos:pos+2] in self.codons_dict[aa]:
                            current_PD_2nd_nt_pos_per_aa_vec[aa][nt2num[ins_nt], pos] = 1
                            
            PD_nt_pos_vec[D_in] = current_PD_nt_pos_vec
            PD_2nd_nt_pos_per_aa_vec[D_in] = current_PD_2nd_nt_pos_per_aa_vec
        
        min_delDl_given_DdelDr = [[]]*num_D_genes
        max_delDl_given_DdelDr = [[]]*num_D_genes
        zeroD_given_D = [[]]*num_D_genes
        for D_in in range(num_D_genes):
            current_min_delDl_given_delDr = [0]*num_delr_pos
            current_max_delDl_given_delDr = [0]*num_delr_pos
            current_zeroD = 0
            for delr in range(num_delr_pos):
                
                if num_dell_pos > len(cutD_genomic_CDR3_segs[D_in])-delr:
                    current_zeroD += generative_model.PdelDldelDr_given_D[len(cutD_genomic_CDR3_segs[D_in])-delr, delr, D_in]
                
                dell = 0
                while generative_model.PdelDldelDr_given_D[dell, delr, D_in]==0 and dell<num_dell_pos-1:
                    dell+=1
                if generative_model.PdelDldelDr_given_D[dell, delr, D_in] == 0:
                    current_min_delDl_given_delDr[delr] = -1
                else:
                    current_min_delDl_given_delDr[delr] = dell
                if current_min_delDl_given_delDr[delr] == -1:
                    current_max_delDl_given_delDr[delr] = -1
                else:
                    dell = num_dell_pos-1
                    while generative_model.PdelDldelDr_given_D[dell, delr, D_in]==0 and dell>=0:
                        dell -= 1
                    if generative_model.PdelDldelDr_given_D[dell, delr, D_in] == 0:
                        current_max_delDl_given_delDr[delr] = -1
                    else:
                        current_max_delDl_given_delDr[delr] = dell
                
            min_delDl_given_DdelDr[D_in] = current_min_delDl_given_delDr
            max_delDl_given_DdelDr[D_in] = current_max_delDl_given_delDr
            zeroD_given_D[D_in] = current_zeroD
        
        self.PD_nt_pos_vec = PD_nt_pos_vec
        self.PD_2nd_nt_pos_per_aa_vec = PD_2nd_nt_pos_per_aa_vec
        self.min_delDl_given_DdelDr = min_delDl_given_DdelDr 
        self.max_delDl_given_DdelDr = max_delDl_given_DdelDr
        self.zeroD_given_D = zeroD_given_D