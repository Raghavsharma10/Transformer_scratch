def ordered(self):
        """An equivalent unit cell with the active cell vectors coming first"""
        active, inactive = self.active_inactive
        order = active + inactive
        return UnitCell(self.matrix[:,order], self.active[order])