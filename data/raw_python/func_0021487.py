def make_V_and_J_mask_mapping(self, genV, genJ):
        """Constructs the V and J mask mapping dictionaries.
        
        Parameters
        ----------
        genV : list
            List of genomic V information.            
        genJ : list
            List of genomic J information.
        
        """
        #construct mapping between allele/gene names and index for custom V_usage_masks
        V_allele_names = [V[0] for V in genV]
        V_mask_mapping = {}
        for v in set([x.split('*')[0] for x in V_allele_names]):
            V_mask_mapping[v] = []
        for v in set(['V'.join((x.split('*')[0]).split('V')[1:]) for x in V_allele_names]):
            V_mask_mapping[v] = []
        for i, v in enumerate(V_allele_names):
            V_mask_mapping[v] = [i]
            V_mask_mapping['V'.join((v.split('*')[0]).split('V')[1:])].append(i)
            V_mask_mapping[v.split('*')[0]].append(i)      
        
        #construct mapping between allele/gene names and index for custom J_usage_masks
        J_allele_names = [J[0] for J in genJ]
        J_mask_mapping = {}
        for j in set([x.split('*')[0] for x in J_allele_names]):
            J_mask_mapping[j] = []
        for j in set(['J'.join((x.split('*')[0]).split('J')[1:]) for x in J_allele_names]):
            J_mask_mapping[j] = []
        for i, j in enumerate(J_allele_names):
            J_mask_mapping[j] = [i]
            J_mask_mapping['J'.join((j.split('*')[0]).split('J')[1:])].append(i)
            J_mask_mapping[j.split('*')[0]].append(i)
            
        self.V_allele_names = V_allele_names
        self.V_mask_mapping = V_mask_mapping
        self.J_allele_names = J_allele_names
        self.J_mask_mapping = J_mask_mapping