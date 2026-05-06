def stringize(
        self,
        rnf_profile,
    ):
        """Create RNF representation of this segment.

		Args:
			rnf_profile (rnftools.rnfformat.RnfProfile): RNF profile (with widths).
		"""

        coor_width = max(rnf_profile.coor_width, len(str(self.left)), len(str(self.right)))
        return "({},{},{},{},{})".format(
            str(self.genome_id).zfill(rnf_profile.genome_id_width),
            str(self.chr_id).zfill(rnf_profile.chr_id_width), self.direction,
            str(self.left).zfill(coor_width),
            str(self.right).zfill(coor_width)
        )