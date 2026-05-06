def next_minifat(self, current):
        """
        Helpers provides access to next mini-FAT sector and returns it's
        seekable position. Should not be called from external code.
        """
        position = 0
        sector_size = self.header.sector_size // 4
        sector = self.header.minifat_sector_start

        while sector != ENDOFCHAIN and (position + 1) * sector_size <= current:
            sector = self.next_fat(sector)
            position += 1

        if sector == ENDOFCHAIN:
            return ENDOFCHAIN

        minifat_position = (sector + 1) << self.header.sector_shift
        minifat_position += (current - position * sector_size) * 4

        return self.get_long(minifat_position)