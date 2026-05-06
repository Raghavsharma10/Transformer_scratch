def load_blind(self, item):
        """Load blind from JSON."""
        blind = Blind.from_config(self.pyvlx, item)
        self.add(blind)