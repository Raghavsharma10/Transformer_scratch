def generate_PVdelV_nt_pos_vecs(self, generative_model, genomic_data):
        """Process P(delV|V) into Pi arrays.
        
        Set the attributes PVdelV_nt_pos_vec and PVdelV_2nd_nt_pos_per_aa_vec.
    
        Parameters
        ----------
        generative_model : GenerativeModelVJ
            VJ generative model class containing the model parameters.            
        genomic_data : GenomicDataVJ
            VJ genomic data class containing the V and J germline 
            sequences and info.
        
        """
    
        cutV_genomic_CDR3_segs = genomic_data.cutV_genomic_CDR3_segs
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num_del_pos = generative_model.PdelV_given_V.shape[0]
        num_V_genes = generative_model.PdelV_given_V.shape[1]
        PVdelV_nt_pos_vec = [[]]*num_V_genes
        PVdelV_2nd_nt_pos_per_aa_vec = [[]]*num_V_genes
        for V_in in range(num_V_genes):
            current_PVdelV_nt_pos_vec = np.zeros((4, len(cutV_genomic_CDR3_segs[V_in])))
            current_PVdelV_2nd_nt_pos_per_aa_vec = {}
            for aa in self.codons_dict.keys():
                current_PVdelV_2nd_nt_pos_per_aa_vec[aa] = np.zeros((4, len(cutV_genomic_CDR3_segs[V_in])))
            for pos, nt in enumerate(cutV_genomic_CDR3_segs[V_in]):
                if len(cutV_genomic_CDR3_segs[V_in]) - pos >  num_del_pos:
                    continue
                if pos%3 == 0: #Start of a codon
                    current_PVdelV_nt_pos_vec[nt2num[nt], pos] = generative_model.PdelV_given_V[len(cutV_genomic_CDR3_segs[V_in])-pos-1, V_in]    
                elif pos%3 == 1: #Mid codon position
                    for ins_nt in 'ACGT':
                        #We need to find what possible codons are allowed for any aa (or motif)
                        for aa in self.codons_dict.keys():
                            if cutV_genomic_CDR3_segs[V_in][pos-1:pos+1]+ ins_nt in self.codons_dict[aa]:
                                current_PVdelV_2nd_nt_pos_per_aa_vec[aa][nt2num[ins_nt], pos] = generative_model.PdelV_given_V[len(cutV_genomic_CDR3_segs[V_in])-pos-1, V_in]            
                elif pos%3 == 2: #End of codon
                    current_PVdelV_nt_pos_vec[0, pos] = generative_model.PdelV_given_V[len(cutV_genomic_CDR3_segs[V_in])-pos-1, V_in]
            PVdelV_nt_pos_vec[V_in] = current_PVdelV_nt_pos_vec
            PVdelV_2nd_nt_pos_per_aa_vec[V_in] = current_PVdelV_2nd_nt_pos_per_aa_vec
    
        
        self.PVdelV_nt_pos_vec = PVdelV_nt_pos_vec
        self.PVdelV_2nd_nt_pos_per_aa_vec = PVdelV_2nd_nt_pos_per_aa_vec