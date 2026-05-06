def prove(self, file, challenge, tag):
        """Returns a proof of ownership of the given file based on the
        challenge.  The proof consists of a hash of the specified file chunk
        and the complete merkle branch.

        :param file: a file that supports `read()`, `seek()` and `tell()`
        :param challenge: the challenge to use for generating this proof
        :param tag: the file tag as provided from the client
        :param filesz: optional filesz parameter.  if not specified, the
            filesz will be detected by seeking to the end of the stream
        """
        leaf = MerkleLeaf(challenge.index,
                          MerkleHelper.get_chunk_hash(file,
                                                      challenge.seed,
                                                      filesz=tag.filesz,
                                                      chunksz=tag.chunksz))
        return Proof(leaf, tag.tree.get_branch(challenge.index))