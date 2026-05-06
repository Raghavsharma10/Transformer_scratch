def destringize(self, string):
        """Get RNF values for this read from its textual representation and save them 
		into this object.

		Args:
			string(str): Textual representation of a read.

		Raises:
			ValueError
		"""

        # todo: assert -- starting with (, ending with )
        # (prefix,read_tuple_id,segments_t,suffix)=(text).split("__")
        # segments=segments_t.split("),(")
        m = read_tuple_destr_pattern.match(string)
        if not m:
            smbl.messages.error(
                "'{}' is not a valid read name with respect to the RNF specification".format(string),
                program="RNFtools", subprogram="RNF format", exception=ValueError
            )
        groups = m.groups()
        # todo: check number of groups
        self.prefix = groups[0]
        read_tuple_id = groups[1]
        self.read_tuple_id = int(read_tuple_id, 16)
        self.segments = []
        segments_str = groups[2:-1]
        for b_str in segments_str:
            if b_str is not None:
                if b_str[0] == ",":
                    b_str = b_str[1:]
                b = rnftools.rnfformat.Segment()
                b.destringize(b_str)
                self.segments.append(b)
        self.suffix = groups[-1]