def generate_PJdelJ_nt_pos_vecs(self, generative_model, genomic_data):
        """Process P(J)*P(delJ|J) into Pi arrays.
        
        Sets the attributes PJdelJ_nt_pos_vec and PJdelJ_2nd_nt_pos_per_aa_vec.
    
        Parameters
        ----------
        generative_model : GenerativeModelVDJ
            VDJ generative model class containing the model parameters.            
        genomic_data : GenomicDataVDJ
            VDJ genomic data class containing the V, D, and J germline 
            sequences and info.
        
        """
    
        cutJ_genomic_CDR3_segs = genomic_data.cutJ_genomic_CDR3_segs
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num_del_pos = generative_model.PdelJ_given_J.shape[0]
    
        num_D_genes, num_J_genes = generative_model.PDJ.shape
        PJ = np.sum(generative_model.PDJ, axis = 0)
    
        PJdelJ_nt_pos_vec = [[]]*num_J_genes
        PJdelJ_2nd_nt_pos_per_aa_vec = [[]]*num_J_genes
        for J_in, pj in enumerate(PJ):
            #We include the marginal PJ here
            current_PJdelJ_nt_pos_vec = np.zeros((4, len(cutJ_genomic_CDR3_segs[J_in])))
            current_PJdelJ_2nd_nt_pos_per_aa_vec  = {}
            for aa in self.codons_dict.keys():
                current_PJdelJ_2nd_nt_pos_per_aa_vec[aa] = np.zeros((4, len(cutJ_genomic_CDR3_segs[J_in])))
    
            for pos, nt in enumerate(cutJ_genomic_CDR3_segs[J_in]):
                if pos >=  num_del_pos:
                    continue
                if (len(cutJ_genomic_CDR3_segs[J_in]) - pos)%3 == 1: #Start of a codon
                    current_PJdelJ_nt_pos_vec[nt2num[nt], pos] = pj*generative_model.PdelJ_given_J[pos, J_in]
                elif (len(cutJ_genomic_CDR3_segs[J_in]) - pos)%3 == 2: #Mid codon position
                    for ins_nt in 'ACGT':
                        #We need to find what possible codons are allowed for any aa (or motif)
                        for aa in self.codons_dict.keys():
                            if ins_nt + cutJ_genomic_CDR3_segs[J_in][pos:pos+2] in self.codons_dict[aa]:
                                current_PJdelJ_2nd_nt_pos_per_aa_vec[aa][nt2num[ins_nt], pos] = pj*generative_model.PdelJ_given_J[pos, J_in]
                                
                elif (len(cutJ_genomic_CDR3_segs[J_in]) - pos)%3 == 0: #End  of codon
                    current_PJdelJ_nt_pos_vec[0, pos] = pj*generative_model.PdelJ_given_J[pos, J_in]
            PJdelJ_nt_pos_vec[J_in] = current_PJdelJ_nt_pos_vec
            PJdelJ_2nd_nt_pos_per_aa_vec[J_in] = current_PJdelJ_2nd_nt_pos_per_aa_vec
        
        self.PJdelJ_nt_pos_vec = PJdelJ_nt_pos_vec
        self.PJdelJ_2nd_nt_pos_per_aa_vec = PJdelJ_2nd_nt_pos_per_aa_vec