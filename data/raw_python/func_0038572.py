def apply(self, read_tuple_name, read_tuple_id=None, synchronize_widths=True):
        """Apply profile on a read tuple name and update read tuple ID.

		Args:
			read_tuple_name (str): Read tuple name to be updated.
			read_tuple_id (id): New read tuple ID.
			synchronize_widths (bool): Update widths (in accordance to this profile).
		"""
        parts = read_tuple_name.split("__")
        parts[0] = self._fill_right(parts[0], "-", self.prefix_width)
        if read_tuple_id is not None:
            parts[1] = "{:x}".format(read_tuple_id)
        parts[1] = self._fill_left(parts[1], "0", self.read_tuple_id_width)

        if synchronize_widths:
            new_segments = []
            segments = parts[2][1:-1].split("),(")
            for segment in segments:
                values = segment.split(",")
                values[0] = values[0].zfill(self.genome_id_width)
                values[1] = values[1].zfill(self.chr_id_width)
                values[3] = values[3].zfill(self.coor_width)
                values[4] = values[4].zfill(self.coor_width)
                new_segments.append("(" + ",".join(values) + ")")
            parts[2] = ",".join(new_segments)

        return "__".join(parts)