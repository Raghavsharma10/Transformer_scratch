def load_igor_genomic_data(self, params_file_name, V_anchor_pos_file, J_anchor_pos_file):
        """Set attributes by loading in genomic data from IGoR parameter file.
        
        Sets attributes genV, genJ, max_delV_palindrome, max_delJ_palindrome,
        cutV_genomic_CDR3_segs, and cutJ_genomic_CDR3_segs.
        
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
        self.genJ = read_igor_J_gene_parameters(params_file_name)
        
        self.anchor_and_curate_genV_and_genJ(V_anchor_pos_file, J_anchor_pos_file)
        
        self.read_igor_VJ_palindrome_parameters(params_file_name)
        
        self.generate_cutV_genomic_CDR3_segs()
        self.generate_cutJ_genomic_CDR3_segs()