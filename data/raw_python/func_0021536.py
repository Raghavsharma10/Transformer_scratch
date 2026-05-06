def anchor_and_curate_genV_and_genJ(self, V_anchor_pos_file, J_anchor_pos_file):
        """Trim V and J germline sequences to the CDR3 region.
        
        Unproductive sequences have an empty string '' for the CDR3 region
        sequence.
        
        Edits the attributes genV and genJ
        
        Parameters
        ----------
        V_anchor_pos_file_name : str
            File name for the conserved residue (C) locations and functionality 
            of each V genomic sequence.
        J_anchor_pos_file_name : str
            File name for the conserved residue (F/W) locations and 
            functionality of each J genomic sequence.
        
        """
        
        V_anchor_pos = load_genomic_CDR3_anchor_pos_and_functionality(V_anchor_pos_file)
        J_anchor_pos = load_genomic_CDR3_anchor_pos_and_functionality(J_anchor_pos_file)
        
        for V in self.genV:
            try:
                if V_anchor_pos[V[0]][0] > 0 and V_anchor_pos[V[0]][1] == 'F': #Check for functionality
                    V[1] = V[2][V_anchor_pos[V[0]][0]:]
                else:
                    V[1] = ''
            except KeyError:
                V[1] = ''
    
        for J in self.genJ:
            try:
                if J_anchor_pos[J[0]][0] > 0 and J_anchor_pos[J[0]][1] == 'F': #Check for functionality
                    J[1] = J[2][:J_anchor_pos[J[0]][0]+3]
                else:
                    J[1] = ''
            except KeyError:
                J[1] = ''