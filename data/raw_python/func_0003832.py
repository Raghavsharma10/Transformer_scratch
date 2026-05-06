def add_molecule(self, molecule, atom_types=None, charges=None, split=True):
        """Add the graph of the molecule to the data structure

           The molecular graph is estimated from the molecular geometry based on
           interatomic distances.

           Argument:
            | ``molecule``  --  a Molecule instance

           Optional arguments:
            | ``atom_types``  --  a list with atom type strings
            | ``charges``  --  The net atom charges
            | ``split``  --  When True, the molecule is split into disconnected
                             molecules [default=True]
        """
        molecular_graph = MolecularGraph.from_geometry(molecule)
        self.add_molecular_graph(molecular_graph, atom_types, charges, split, molecule)