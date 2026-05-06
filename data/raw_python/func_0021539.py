def load_igor_genomic_data(self, params_file_name, V_anchor_pos_file, J_anchor_pos_file):
        """Set attributes by loading in genomic data from IGoR parameter file.
        
        Sets attributes genV,  max_delV_palindrome, cutV_genomic_CDR3_segs, 
        genD, max_delDl_palindrome, max_delDr_palindrome, 
        cutD_genomic_CDR3_segs, genJ, max_delJ_palindrome, and 
        cutJ_genomic_CDR3_segs.
        
        Parameters
        ----------
        params_file_name : str
            File name for a IGOR parameter file.
        V_anchor_pos_file_name : str
            File name for the conserved residue (C) locations and functionality 
            of each V genomic sequence.
        J_anchor_pos_file_name : str
            File name for the conserved residue (F/W) locations and 
            functionality of each J genomic sequence.
        
        """
        
        self.genV = read_igor_V_gene_parameters(params_file_name)
        self.genD = read_igor_D_gene_parameters(params_file_name)
        self.genJ = read_igor_J_gene_parameters(params_file_name)
        
        self.anchor_and_curate_genV_and_genJ(V_anchor_pos_file, J_anchor_pos_file)
 
        self.read_VDJ_palindrome_parameters(params_file_name) #Need palindrome info before generating cut_genomic_CDR3_segs

        self.generate_cutV_genomic_CDR3_segs()
        self.generate_cutD_genomic_CDR3_segs()
        self.generate_cutJ_genomic_CDR3_segs()