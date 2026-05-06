def verify(self, proof, challenge, state):
        """returns true if the proof matches the challenge.  verifies that the
        server possesses the encoded file.

        :param proof: the proof that was returned from the server
        :param challenge: the challenge provided to the server
        :param state: the state of the file, which includes the merkle root of
        of the merkle tree, for verification.
        """
        state.checksig(self.key)
        if (proof.leaf.index != challenge.index):
            return False
        return MerkleTree.verify_branch(proof.leaf,
                                        proof.branch,
                                        state.root)