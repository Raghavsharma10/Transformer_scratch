def get_hex_chain(self, index):
        """Assemble and return the chain leading from a given node to the merkle root of this tree
        with hash values in hex form
        """
        return [(codecs.encode(i[0], 'hex_codec'), i[1]) for i in self.get_chain(index)]