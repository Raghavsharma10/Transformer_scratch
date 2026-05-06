def modelstandardfunctions(self):
        """Standard functions of the model class."""
        lines = Lines()
        lines.extend(self.doit)
        lines.extend(self.iofunctions)
        lines.extend(self.new2old)
        lines.extend(self.run)
        lines.extend(self.update_inlets)
        lines.extend(self.update_outlets)
        lines.extend(self.update_receivers)
        lines.extend(self.update_senders)
        return lines