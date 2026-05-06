def combine(*rnf_profiles):
        """Combine more profiles and set their maximal values.

		Args:
			*rnf_profiles (rnftools.rnfformat.RnfProfile): RNF profile.
		"""

        for rnf_profile in rnf_profiles:
            self.prefix_width = max(self.prefix_width, rnf_profile.prefix_width)
            self.read_tuple_id_width = max(self.read_tuple_id_width, rnf_profile.read_tuple_id_width)
            self.genome_id_width = max(self.genome_id_width, rnf_profile.genome_id_width)
            self.chr_id_width = max(self.chr_id_width, rnf_profile.chr_id_width)
            self.coor_width = max(self.coor_width, rnf_profile.coor_width)