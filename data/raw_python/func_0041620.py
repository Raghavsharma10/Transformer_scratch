def _structure(self, source_code):
        """return structure in ACDP format."""
        # define cutter as a per block reader
        def cutter(seq, block_size):
            for index in range(0, len(seq), block_size):
                lexem = seq[index:index+block_size]
                if len(lexem) == block_size:
                    yield self.table_struct[seq[index:index+block_size]]
        return tuple(cutter(source_code, self.idnt_struct_size))