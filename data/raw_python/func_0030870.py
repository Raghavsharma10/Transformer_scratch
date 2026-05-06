def add_hash(self, value):
        """Add a Node based on a precomputed, hex encoded, hash value.
        """
        self.leaves.append(Node(codecs.decode(value, 'hex_codec'), prehashed=True))