def check(self, read_tuple_name):
        """Check if the given read tuple name satisfies this profile.

		Args:
			read_tuple_name (str): Read tuple name.
		"""

        parts = read_tuple_name.split("__")

        if len(parts[0]) != self.prefix_width or len(parts[1]) != self.read_tuple_id_width:
            return False

        segments = parts[2][1:-1].split("),(")
        for segment in segments:
            int_widths = list(map(len, segment.split(",")))
            if self.genome_id_width != int_widths[0]:
                return False
            if self.chr_id_width != int_widths[1]:
                return False
            if self.coor_width != int_widths[3] or self.coor_width != int_widths[4]:
                return False

        return True