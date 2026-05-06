def next_fat(self, current):
        """
        Helper gives you seekable position of next FAT sector. Should not be
        called from external code.
        """
        sector_size = self.header.sector_size // 4
        block = current // sector_size
        difat_position = 76

        if block >= 109:
            block -= 109
            sector = self.header.difat_sector_start

            while block >= sector_size:
                position = (sector + 1) << self.header.sector_shift
                position += self.header.sector_size - 4
                sector = self.get_long(position)
                block -= sector_size - 1

            difat_position = (sector + 1) << self.header.sector_shift
        fat_sector = self.get_long(difat_position + block * 4)

        fat_position = (fat_sector + 1) << self.header.sector_shift
        fat_position += (current % sector_size) * 4

        return self.get_long(fat_position)