def stringize(
        self,
        rnf_profile=RnfProfile(),
    ):
        """Create RNF representation of this read.

		Args:
			read_tuple_id_width (int): Maximal expected string length of read tuple ID.
			genome_id_width (int): Maximal expected string length of genome ID.
			chr_id_width (int): Maximal expected string length of chromosome ID.
			coor_width (int): Maximal expected string length of a coordinate.
		"""

        sorted_segments = sorted(self.segments,
         key=lambda x: (
          x.genome_id * (10 ** 23) +
          x.chr_id * (10 ** 21) +
          (x.left + (int(x.left == 0) * x.right - 1)) * (10 ** 11) +
          x.right * (10 ** 1) +
          int(x.direction == "F")
         )
        )

        segments_strings = [x.stringize(rnf_profile) for x in sorted_segments]

        read_tuple_name = "__".join(
            [
                self.prefix,
                format(self.read_tuple_id, 'x').zfill(rnf_profile.read_tuple_id_width),
                ",".join(segments_strings),
                self.suffix,
            ]
        )

        return read_tuple_name