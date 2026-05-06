def root(self):
        """ Property provides access to root object in CFB. """
        sector = self.header.directory_sector_start
        position = (sector + 1) << self.header.sector_shift
        return RootEntry(self, position)