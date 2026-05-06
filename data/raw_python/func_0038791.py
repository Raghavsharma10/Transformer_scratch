def destringize(self, string):
        """Get RNF values for this segment from its textual representation and
		save them into this object.

		Args:
			string (str): Textual representation of a segment.
		"""

        m = segment_destr_pattern.match(string)
        self.genome_id = int(m.group(1))
        self.chr_id = int(m.group(2))
        self.direction = m.group(3)
        self.left = int(m.group(4))
        self.right = int(m.group(5))