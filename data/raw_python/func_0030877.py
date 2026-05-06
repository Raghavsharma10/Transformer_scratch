def get_all_hex_chains(self):
        """Assemble and return a list of all chains for all nodes to the merkle root, hex encoded.
        """
        return [[(codecs.encode(i[0], 'hex_codec'), i[1]) for i in j] for j in self.get_all_chains()]