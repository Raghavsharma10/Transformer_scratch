def set_bio_assembly_trans(self, bio_assembly_index, input_chain_indices, input_transform):
        """Set the Bioassembly transformation information. A single bioassembly can have multiple transforms,
        :param bio_assembly_index: the integer index of the bioassembly
        :param input_chain_indices: the list of integer indices for the chains of this bioassembly
        :param input_transformation: the list of doubles for  the transform of this bioassmbly transform"""
        this_bioass = None
        for bioass in self.bio_assembly:
            if bioass['name'] == str(bio_assembly_index):
                this_bioass = bioass
                break
        if not this_bioass:
            this_bioass = {"name": str(bio_assembly_index), 'transformList': []}
        else:
            self.bio_assembly.remove(this_bioass)
        this_bioass['transformList'].append({'chainIndexList':input_chain_indices,'matrix': input_transform})
        self.bio_assembly.append(this_bioass)