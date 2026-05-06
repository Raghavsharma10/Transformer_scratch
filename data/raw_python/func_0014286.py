def write_pdb(self, path):
        """
        Outputs a PDB file with the current contents of the system
        """
        if self.master is None and self.positions is None:
            raise ValueError('Topology and positions are needed to write output files.')
        with open(path, 'w') as f:
            PDBFile.writeFile(self.topology, self.positions, f)