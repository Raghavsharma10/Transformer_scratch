def add_read(
        self,
        read_tuple_id,
        bases,
        qualities,
        segments,
    ):
        """Add a new read to the current buffer. If it is a new read tuple (detected from ID), the buffer will be flushed.

		Args:
			read_tuple_id (int): ID of the read tuple.
			bases (str): Sequence of bases.
			qualities (str): Sequence of FASTQ qualities.
			segments (list of rnftools.rnfformat.segment): List of segments constituting the read.
		"""

        assert type(bases) is str, "Wrong type of bases: '{}'".format(bases)
        assert type(qualities) is str, "Wrong type of qualities: '{}'".format(qualities)
        assert type(segments) is tuple or type(segments) is list

        if self.current_read_tuple_id != read_tuple_id:
            self.flush_read_tuple()
        self.current_read_tuple_id = read_tuple_id

        self.seqs_bases.append(bases)
        self.seqs_qualities.append(qualities)
        self.segments.extend(segments)