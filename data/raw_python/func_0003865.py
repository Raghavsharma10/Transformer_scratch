def set_default_masses(self):
        """Set self.masses based on self.numbers and periodic table."""
        self.masses = np.array([periodic[n].mass for n in self.numbers])