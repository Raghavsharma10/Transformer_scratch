def load(self, read_tuple_name):
        """Load RNF values from a read tuple name.

		Args:
			read_tuple_name (str): Read tuple name which the values are taken from.
		"""
        self.prefix_width = 0
        self.read_tuple_id_width = 0
        self.genome_id_width = 0
        self.chr_id_width = 0
        self.coor_width = 0

        parts = read_tuple_name.split("__")
        self.prefix_width = len(parts[0])
        self.read_tuple_id_width = len(parts[1])

        segments = parts[2][1:-1].split("),(")
        for segment in segments:
            int_widths = list(map(len, segment.split(",")))
            self.genome_id_width = max(self.genome_id_width, int_widths[0])
            self.chr_id_width = max(self.chr_id_width, int_widths[1])
            self.coor_width = max(self.coor_width, int_widths[2], int_widths[3])