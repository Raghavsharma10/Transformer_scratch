def init_structure(self, total_num_bonds, total_num_atoms,
                       total_num_groups, total_num_chains, total_num_models,
                       structure_id):
        """Initialise the structure object.
        :param total_num_bonds: the number of bonds in the structure
        :param total_num_atoms: the number of atoms in the structure
        :param total_num_groups: the number of groups in the structure
        :param total_num_chains: the number of chains in the structure
        :param total_num_models: the number of models in the structure
        :param structure_id the: id of the structure (e.g. PDB id)
        """
        self.mmtf_version = constants.MMTF_VERSION
        self.mmtf_producer = constants.PRODUCER
        self.num_atoms = total_num_atoms
        self.num_bonds = total_num_bonds
        self.num_groups = total_num_groups
        self.num_chains = total_num_chains
        self.num_models = total_num_models
        self.structure_id = structure_id
        # initialise the arrays
        self.x_coord_list = []
        self.y_coord_list = []
        self.z_coord_list = []
        self.group_type_list = []
        self.entity_list = []
        self.b_factor_list = []
        self.occupancy_list = []
        self.atom_id_list = []
        self.alt_loc_list = []
        self.ins_code_list = []
        self.group_id_list = []
        self.sequence_index_list = []
        self.group_list = []
        self.chain_name_list = []
        self.chain_id_list = []
        self.bond_atom_list = []
        self.bond_order_list = []
        self.sec_struct_list = []
        self.chains_per_model = []
        self.groups_per_chain = []
        self.current_group = None
        self.bio_assembly = []