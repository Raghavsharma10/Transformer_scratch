def encode(self,
               file,
               n=DEFAULT_CHALLENGE_COUNT,
               seed=None,
               chunksz=None,
               filesz=None):
        """This function generates a merkle tree with the leaves as seed file
        hashes, the seed for each leaf being a deterministic seed generated
        from a key.

        :param file: a file like object that supports the `read()`, `seek()`
        and `tell()` methods
        :param n: the number of challenges to generate
        :param seed: the root seed for this batch of challenges.  by default
            generates a random seed
        :param chunksz: the chunk size for breaking up the file: the amount
            of the file that will be checked by each challenge.  defaults
            to the chunk size defined by check_fraction
        :param filesz: optional size of the file.  if not specified, file size
            will be detected by seeking to the end of the file and reading the
            position
        """
        if (seed is None):
            seed = os.urandom(DEFAULT_KEY_SIZE)
        if (filesz is None):
            file.seek(0, 2)
            filesz = file.tell()
        if (chunksz is None):
            if (self.check_fraction is not None):
                chunksz = int(self.check_fraction * filesz)
            else:
                chunksz = DEFAULT_CHUNK_SIZE
        mt = MerkleTree()
        state = State(0, seed, n)
        seed = MerkleHelper.get_next_seed(self.key, state.seed)
        for i in range(0, n):
            leaf = MerkleHelper.get_chunk_hash(file, seed, filesz, chunksz)
            mt.add_leaf(leaf)
            seed = MerkleHelper.get_next_seed(self.key, seed)
        mt.build()
        state.root = mt.get_root()
        mt.strip_leaves()
        tag = Tag(mt, chunksz, filesz)
        state.sign(self.key)
        return (tag, state)