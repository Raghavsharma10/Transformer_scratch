def clear(self):
        """Clear the contents of the data structure"""
        self.title = None
        self.numbers = np.zeros(0, int)
        self.atom_types = [] # the atom_types in the second column, used to associate ff parameters
        self.charges = [] # ff charges
        self.names = [] # a name that is unique for the molecule composition and connectivity
        self.molecules = np.zeros(0, int) # a counter for each molecule
        self.bonds = np.zeros((0, 2), int)
        self.bends = np.zeros((0, 3), int)
        self.dihedrals = np.zeros((0, 4), int)
        self.impropers = np.zeros((0, 4), int)

        self.name_cache = {}